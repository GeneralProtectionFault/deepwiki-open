from typing import Sequence, List, Optional, Callable  
from copy import deepcopy  
from tqdm import tqdm  
import logging  
import re  
import adalflow as adal  
from adalflow.core.types import Document  
from adalflow.core.component import DataComponent  
import requests  
import os  
import copyreg  
import types  
import importlib  

import tiktoken  
from ollama import ResponseError as OllamaResponseError  

# Configure logging
from api.logging_config import setup_logging
from adalflow.components.model_client.ollama_client import OllamaClient as _AdalflowOllamaClient

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pickling helpers (unchanged)
# ---------------------------------------------------------------------------

def _pickle_module(module):
    return (_unpickle_module, (module.__name__,))

def _unpickle_module(name):
    return importlib.import_module(name)

copyreg.pickle(types.ModuleType, _pickle_module)


# ---------------------------------------------------------------------------
# Token budget helpers
# ---------------------------------------------------------------------------

# Configurable max tokens per single embedding API call.  Override via
# environment variable DEEPWIKI_EMBED_CHUNK_LIMIT.  Ollama's *actual*
# physical batch size is often much smaller than the advertised num_ctx,
# so the default is deliberately conservative.
def _get_embed_chunk_limit() -> int:
    """
    Read the per-input token limit for Ollama embeddings from:
    1. Environment variable DEEPWIKI_EMBED_CHUNK_LIMIT
    2. embedder_ollama.embed_chunk_limit in embedder.json
    3. Hard-coded default (1500)
    """
    # Env var takes highest precedence
    env_val = os.environ.get("DEEPWIKI_EMBED_CHUNK_LIMIT")
    if env_val is not None:
        try:
            return max(1, int(env_val))
        except (TypeError, ValueError):
            pass

    # Fall back to config file
    try:
        from api.config import get_embedder_config
        embedder_config = get_embedder_config()
        cfg_limit = embedder_config.get("embed_chunk_limit")
        if cfg_limit is not None:
            return max(1, int(cfg_limit))
    except Exception as e:
        logger.debug(f"Could not read embed_chunk_limit from config: {e}")

    return 1500


def _get_safe_embed_token_budget(default: int = 8192, safety_margin: float = 0.9) -> int:
    """
    Read num_ctx from the active Ollama embedder config and apply a safety margin.
    Falls back to `default` if config lookup fails for any reason.
    Kept for backward compatibility.
    """
    try:
        from api.config import get_embedder_config
        embedder_config = get_embedder_config()
        num_ctx = (
            embedder_config.get("model_kwargs", {})
            .get("options", {})
            .get("num_ctx", default)
        )
        return max(1, int(num_ctx * safety_margin))
    except Exception as e:
        logger.warning(f"Could not read num_ctx from embedder config, using default {default}: {e}")
        return int(default * safety_margin)


# ---------------------------------------------------------------------------
# Chunking helpers  (new)
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Fast token-count estimate using tiktoken cl100k_base."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def _chunk_text(text: str, max_tokens: int) -> List[str]:
    """
    Split *text* into substrings that each fit within *max_tokens* tokens.

    Tries paragraph-level splits first, then line-level, then a raw
    token-level fallback.  Every returned chunk is guaranteed to be
    <= max_tokens (estimated) except possibly the last fallback piece.
    """
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    # Strategy 1 -- paragraph-level
    paragraphs = re.split(r'\n\s*\n', text)
    result = _chunk_by_units(paragraphs, max_tokens)
    if result:
        return result

    # Strategy 2 -- line-level
    lines = text.split('\n')
    result = _chunk_by_units(lines, max_tokens)
    if result:
        return result

    # Strategy 3 -- hard token-level split
    return _chunk_by_tokens(text, max_tokens)


def _chunk_by_units(units: List[str], max_tokens: int) -> List[str]:
    """
    Greedily pack *units* (paragraphs, lines, …) into chunks each <= *max_tokens*.
    """
    if not units:
        return []

    chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = _estimate_tokens(unit)
        if current_tokens + unit_tokens <= max_tokens:
            current_parts.append(unit)
            current_tokens += unit_tokens
        else:
            if current_parts:
                chunks.append('\n'.join(current_parts))
            if unit_tokens > max_tokens:
                # Single unit still too big -- emit it; caller falls through
                chunks.append(unit)
            current_parts = [unit]
            current_tokens = unit_tokens

    if current_parts:
        chunks.append('\n'.join(current_parts))

    return chunks


def _chunk_by_tokens(text: str, max_tokens: int) -> List[str]:
    """
    Hard split *text* into chunks of at most *max_tokens* tokens each.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        chunks: List[str] = []
        for start in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[start:start + max_tokens]
            chunks.append(encoding.decode(chunk_tokens))
        return chunks
    except Exception as e:
        logger.warning(f"Token-level chunking failed: {e}")
        return [text]


def _mean_pool_embeddings(embeddings: List[List[float]]) -> List[float]:
    """
    Compute the element-wise mean of a list of embedding vectors of
    identical dimension.
    """
    if not embeddings:
        raise ValueError("Cannot mean-pool an empty list of embeddings")
    dim = len(embeddings[0])
    pooled = [0.0] * dim
    count = len(embeddings)
    for emb in embeddings:
        if len(emb) != dim:
            logger.warning("Embedding dimension mismatch: expected %d, got %d -- skipping",
                          dim, len(emb))
            continue
        for j in range(dim):
            pooled[j] += emb[j]
    return [v / count for v in pooled]


# ---------------------------------------------------------------------------
# Backward-compat helpers (kept for external callers)
# ---------------------------------------------------------------------------

def _truncate_text_to_token_budget(text: str, token_budget: int) -> str:
    """Truncate text to *token_budget* tokens.  Kept for backward compat."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) <= token_budget:
            return text
        return encoding.decode(tokens[:token_budget])
    except Exception as e:
        logger.warning(f"Token truncation failed, falling back to char-based estimate: {e}")
        return text[:token_budget * 4]


def _is_context_length_error(exc: Exception) -> bool:
    if isinstance(exc, OllamaResponseError):
        return True
    return "input length exceeds the context length" in str(exc).lower()


# ---------------------------------------------------------------------------
# Ollama model checker
# ---------------------------------------------------------------------------

class OllamaModelNotFoundError(Exception):
    """Custom exception for when Ollama model is not found"""
    pass


def check_ollama_model_exists(model_name: str, ollama_host: str = None) -> bool:
    """Check if an Ollama model exists before attempting to use it."""
    if ollama_host is None:
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    else:
        print(f"Current (unmodified from env) Ollama host: {ollama_host}")
    try:
        if ollama_host.endswith('/api'):
            ollama_host = ollama_host[:-4]

        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model.get('name', '') for model in models_data.get('models', [])]
            is_available = model_name in available_models or model_name.split(':')[0] in [m.split(':')[0] for m in available_models]
            if is_available:
                logger.info(f"Ollama model '{model_name}' is available")
            else:
                logger.warning(f"Ollama model '{model_name}' is not available. Available models: {available_models}")
            return is_available
        else:
            logger.warning(f"Could not check Ollama models, status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not connect to Ollama to check models: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error checking Ollama model availability: {e}")
        return False


# ---------------------------------------------------------------------------
# Ollama document processor -- chunk-then-embed
# ---------------------------------------------------------------------------

class OllamaDocumentProcessor(DataComponent):
    """
    Process documents for Ollama embeddings.  Each document is embedded
    individually (Ollama does not support batch embedding).  When a
    document exceeds the per-input token limit set by
    ``DEEPWIKI_EMBED_CHUNK_LIMIT`` it is split into safe-sized chunks,
    each chunk is embedded separately, and the chunk embeddings are
    mean-pooled into a single vector for the document.  No content is
    dropped.
    """
    def __init__(
        self,
        embedder: adal.Embedder,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.progress_callback = progress_callback

    def __reduce__(self):
        # LocalDB.save_state() pickles this whole component. The embedder's
        # model_client chain cannot be safely pickled. By the time
        # save_state() runs, __call__ has already computed and attached
        # .vector to every Document, so the embedder is not needed again.
        return (self.__class__, (None,))

    # -- inner helper ----------------------------------------------------------

    def _embed_with_chunking(self, text: str) -> Optional[List[float]]:
        """
        Embed *text*.  If it fits in one call the embedder is invoked once;
        otherwise split into safe-sized chunks, embed each, and mean-pool.
        """
        chunk_limit = _get_embed_chunk_limit()
        chunks = _chunk_text(text, chunk_limit)

        if len(chunks) == 1:
            result = self.embedder(input=text)
            if result and result.data and len(result.data) > 0:
                return result.data[0].embedding
            return None

        logger.info(
            "Chunking oversized text into %d pieces for embedding (limit=%d tokens)",
            len(chunks), chunk_limit,
        )
        chunk_embeddings: List[List[float]] = []
        for ci, chunk in enumerate(chunks):
            cr = self.embedder(input=chunk)
            if cr and cr.data and len(cr.data) > 0:
                chunk_embeddings.append(cr.data[0].embedding)
            else:
                logger.warning("Empty embedding returned for chunk %d/%d -- skipping",
                              ci + 1, len(chunks))
        if not chunk_embeddings:
            return None
        return _mean_pool_embeddings(chunk_embeddings)

    # -- public API -----------------------------------------------------------

    def __call__(self, documents: Sequence[Document]) -> Sequence[Document]:
        output = deepcopy(documents)
        logger.info(f"Processing {len(output)} documents individually for Ollama embeddings")

        successful_docs: List[Document] = []
        expected_embedding_size: Optional[int] = None

        for i, doc in enumerate(tqdm(output, desc="Processing documents for Ollama embeddings")):
            file_path = getattr(doc, "meta_data", {}).get("file_path", f"document_{i}")
            try:
                embedding = self._embed_with_chunking(doc.text)
                if embedding is None:
                    logger.warning("Failed to get embedding for '%s', skipping", file_path)
                    continue

                if expected_embedding_size is None:
                    expected_embedding_size = len(embedding)
                elif len(embedding) != expected_embedding_size:
                    logger.warning(
                        "Document '%s' has inconsistent embedding size %d != %d, skipping",
                        file_path, len(embedding), expected_embedding_size,
                    )
                    continue

                output[i].vector = embedding
                successful_docs.append(output[i])
            except Exception as e:
                logger.error("Error processing document '%s': %s, skipping", file_path, e)

            if self.progress_callback:
                self.progress_callback(i + 1, len(output))

        logger.info(
            "Successfully processed %d/%d documents with consistent embeddings",
            len(successful_docs), len(output),
        )
        self.progress_callback = None
        return successful_docs
