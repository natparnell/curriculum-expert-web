#!/usr/bin/env python3
"""
RAG Pipeline for Curriculum Knowledge Base
Indexes markdown files from knowledge/{subject}-curriculum/ directories
into ChromaDB for semantic search and retrieval.
Subjects are loaded dynamically from curriculum-agent-config.json.
"""

import os
import sys
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib

import chromadb
from chromadb.config import Settings
import openai

# Configuration
CLOUD_MODE = os.environ.get('CLOUD_MODE', '').lower() in ('1', 'true', 'yes')
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/app"))
KNOWLEDGE_DIR = Path(os.environ.get("KNOWLEDGE_DIR", str(WORKSPACE / "knowledge")))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", str(WORKSPACE / "data" / "chromadb")))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_MIN_WORDS = 100
CHUNK_MAX_WORDS = 600
CONFIG_PATH = Path(__file__).parent / "curriculum-agent-config.json"

def _load_subjects_from_config() -> List[str]:
    """Read subject keys from the central config file."""
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        return [k for k in config.get("subjects", {}).keys()
                if not k.startswith("_")]
    except Exception:
        return ["history", "geography"]  # safe fallback

# Module-level singleton — initialized lazily on first use
_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        key = OPENAI_API_KEY
        if not key and not CLOUD_MODE:
            try:
                with open('/home/node/.openclaw/clawdbot.json') as f:
                    cj = json.load(f)
                key = cj.get('env', {}).get('vars', {}).get('OPENAI_API_KEY')
            except Exception:
                pass
        _openai_client = openai.OpenAI(api_key=key)
    return _openai_client

class RAGPipeline:
    def __init__(self):
        self.client = None
        self.collection = None
        self._init_chromadb()
        
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="curriculum",
            metadata={"description": "Curriculum knowledge base"}
        )
        
    def chunk_markdown_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """
        Split markdown file into chunks based on H2/H3 headings.
        Returns list of {text, metadata} dicts.
        """
        content = filepath.read_text(encoding='utf-8')
        
        # Split on H2 (## ) and H3 (### ) headings
        # Pattern matches lines starting with ## or ###
        pattern = r'^(#{2,3}\s+.+)$'
        parts = re.split(pattern, content, flags=re.MULTILINE)
        
        chunks = []
        current_heading = "Introduction"
        current_content = []
        
        for part in parts:
            if not part.strip():
                continue
                
            if part.startswith('## ') or part.startswith('### '):
                # Save previous chunk if it has content
                if current_content:
                    chunk_text = '\n'.join(current_content).strip()
                    word_count = len(chunk_text.split())
                    
                    if word_count >= CHUNK_MIN_WORDS:
                        chunks.append({
                            'text': chunk_text,
                            'heading': current_heading,
                            'word_count': word_count
                        })
                    elif word_count > 20:  # Keep smaller chunks but skip tiny stubs
                        chunks.append({
                            'text': chunk_text,
                            'heading': current_heading,
                            'word_count': word_count
                        })
                
                # Start new chunk
                current_heading = part.strip()
                current_content = []
            else:
                current_content.append(part)
        
        # Don't forget the last chunk
        if current_content:
            chunk_text = '\n'.join(current_content).strip()
            word_count = len(chunk_text.split())
            
            if word_count >= 20:  # Keep if meaningful
                chunks.append({
                    'text': chunk_text,
                    'heading': current_heading,
                    'word_count': word_count
                })
        
        # Split oversized chunks
        final_chunks = []
        for chunk in chunks:
            if chunk['word_count'] > CHUNK_MAX_WORDS:
                # Split in half
                words = chunk['text'].split()
                mid = len(words) // 2
                
                # Find a good break point (end of sentence)
                text1 = ' '.join(words[:mid])
                text2 = ' '.join(words[mid:])
                
                final_chunks.append({
                    'text': text1,
                    'heading': chunk['heading'],
                    'word_count': len(text1.split())
                })
                final_chunks.append({
                    'text': text2,
                    'heading': chunk['heading'] + " (continued)",
                    'word_count': len(text2.split())
                })
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def build_metadata(self, filepath: Path, heading: str) -> Dict[str, str]:
        """Build metadata dict for a chunk."""
        path_str = str(filepath.relative_to(WORKSPACE))
        
        # Infer subject from path dynamically
        subject = "unknown"
        for subj in _load_subjects_from_config():
            if f"{subj}-curriculum" in path_str:
                subject = subj
                break
        
        # Infer key stage from path/heading
        key_stage = "general"
        if "eyfs" in path_str.lower() or "eyfs" in heading.lower():
            key_stage = "eyfs"
        elif "ks1" in path_str.lower() or "ks1" in heading.lower():
            key_stage = "ks1"
        elif "ks2" in path_str.lower() or "ks2" in heading.lower():
            key_stage = "ks2"
        elif "ks3" in path_str.lower() or "ks3" in heading.lower():
            key_stage = "ks3"
        elif "ks4" in path_str.lower() or "ks4" in heading.lower() or "gcse" in heading.lower():
            key_stage = "ks4"
        elif "a_level" in path_str.lower() or "a-level" in heading.lower():
            key_stage = "a_level"
        
        # Infer thinker if file is in key_thinkers folder
        thinker = None
        if "key_thinkers" in path_str:
            # Extract thinker name from filename (e.g., christine_counsell.md)
            filename = filepath.stem.replace('_', ' ').title()
            thinker = filename
        
        metadata = {
            "subject": subject,
            "file_path": path_str,
            "key_stage": key_stage,
            "heading": heading,
            "indexed_at": datetime.utcnow().isoformat() + "Z"
        }
        
        if thinker:
            metadata["thinker"] = thinker
            
        return metadata
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API."""
        client = _get_openai_client()
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text[:8000]  # OpenAI has token limits
        )
        return response.data[0].embedding
    
    def _file_content_hash(self, filepath: Path) -> str:
        """MD5 of file contents — used to detect changes and skip unchanged files."""
        return hashlib.md5(filepath.read_bytes()).hexdigest()

    def _get_indexed_hashes(self) -> dict:
        """Return {file_path_str: content_hash} for all chunks currently in ChromaDB."""
        try:
            results = self.collection.get(include=["metadatas"])
            hashes = {}
            for meta in results.get("metadatas", []):
                fp = meta.get("file_path")
                ch = meta.get("content_hash")
                if fp and ch:
                    hashes[fp] = ch  # last write wins — all chunks from same file share same hash
            return hashes
        except Exception:
            return {}

    def index_knowledge_base(self, subjects: List[str] = None, force: bool = False) -> Dict[str, int]:
        """
        Index all knowledge files for specified subjects.
        Skips files whose content hash matches what's already in ChromaDB (incremental).
        Pass force=True to re-embed everything regardless.
        Returns dict with counts.
        """
        if subjects is None:
            subjects = _load_subjects_from_config()

        indexed_hashes = {} if force else self._get_indexed_hashes()

        total_chunks = 0
        total_files = 0
        skipped_files = 0

        for subject in subjects:
            subject_dir = KNOWLEDGE_DIR / f"{subject}-curriculum"
            if not subject_dir.exists():
                print(f"Directory not found: {subject_dir}")
                continue

            # Find all markdown files
            md_files = list(subject_dir.rglob("*.md"))

            for filepath in md_files:
                # Skip index and build files
                if filepath.name in ["00_INDEX.md", "BUILD_QUEUE.md", "PROGRESS.md", "cross-agent-log.md"]:
                    continue

                # Skip memory directory
                if "memory" in str(filepath):
                    continue

                # Skip if content unchanged
                rel_path = str(filepath.relative_to(WORKSPACE))
                current_hash = self._file_content_hash(filepath)
                if not force and indexed_hashes.get(rel_path) == current_hash:
                    print(f"Skipping (unchanged): {rel_path}")
                    skipped_files += 1
                    continue

                print(f"Indexing: {filepath.relative_to(WORKSPACE)}")
                
                # Chunk the file
                chunks = self.chunk_markdown_file(filepath)
                
                if not chunks:
                    continue
                
                total_files += 1
                
                # Prepare batch for embedding
                texts = []
                metadatas = []
                ids = []
                
                for i, chunk in enumerate(chunks):
                    chunk_text = chunk['text']
                    metadata = self.build_metadata(filepath, chunk['heading'])
                    metadata['content_hash'] = current_hash

                    # Create unique ID
                    id_str = hashlib.md5(f"{filepath}:{chunk['heading']}:{i}".encode()).hexdigest()
                    
                    texts.append(chunk_text)
                    metadatas.append(metadata)
                    ids.append(id_str)
                
                # Get embeddings in batch (OpenAI supports up to 2048 per batch)
                embeddings = []
                batch_size = 50  # Conservative batch size
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    client = _get_openai_client()
                    response = client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        input=batch_texts
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                
                # Upsert to ChromaDB
                self.collection.upsert(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                
                total_chunks += len(chunks)
                print(f"  → Indexed {len(chunks)} chunks")
        
        result = {
            "files_indexed": total_files,
            "files_skipped": skipped_files,
            "chunks_indexed": total_chunks,
            "subjects": subjects,
            "indexed_at": datetime.utcnow().isoformat() + "Z"
        }
        
        # Log to cross-agent-log
        self._log_indexing(result)
        
        return result
    
    def _log_indexing(self, result: Dict):
        """Log indexing results to cross-agent-log.md (skipped in cloud mode)."""
        if CLOUD_MODE:
            return
        log_path = WORKSPACE / "memory" / "cross-agent-log.md"
        log_entry = f"""
## {datetime.utcnow().isoformat()} — RAG Indexing

**Files indexed:** {result['files_indexed']}
**Chunks indexed:** {result['chunks_indexed']}
**Subjects:** {', '.join(result['subjects'])}

"""
        try:
            if log_path.exists():
                content = log_path.read_text()
                log_path.write_text(log_entry + content)
            else:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(log_entry)
        except Exception:
            pass  # Non-fatal
    
    def query(self, question: str, subject: Optional[str] = None, top_k: int = 5, max_per_source: int = 2) -> List[Dict[str, Any]]:
        """
        Query the knowledge base.
        Returns list of {text, metadata, distance} dicts.
        
        max_per_source: cap on chunks returned from any single file.
        Prevents one large file (e.g. a detailed thinker profile) from
        dominating all top-k slots. Defaults to 2.
        """
        # Get embedding for question
        question_embedding = self.get_embedding(question)
        
        # Build where filter if subject specified
        where_filter = None
        if subject:
            where_filter = {"subject": subject}
        
        # Fetch more results than needed so we have room to deduplicate by source.
        # Fetch top_k * max_per_source * 2 to give enough candidates.
        fetch_n = min(top_k * max_per_source * 2, 50)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=fetch_n,
            where=where_filter
        )
        
        # Format all candidates
        candidates = []
        for i in range(len(results['ids'][0])):
            candidates.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        # Deduplicate: cap chunks per source file, preserve relevance order
        source_counts: Dict[str, int] = {}
        formatted = []
        for chunk in candidates:
            source = chunk['metadata'].get('file_path', 'unknown')
            count = source_counts.get(source, 0)
            if count < max_per_source:
                formatted.append(chunk)
                source_counts[source] = count + 1
            if len(formatted) >= top_k:
                break
        
        return formatted
    
    def format_for_prompt(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks for inclusion in LLM prompt."""
        lines = ["The following excerpts are from the curriculum knowledge base:"]
        lines.append("")
        
        for i, chunk in enumerate(chunks, 1):
            meta = chunk['metadata']
            source = f"{meta.get('file_path', 'Unknown')} → {meta.get('heading', 'Unknown')}"
            
            lines.append(f"[{i}] Source: {source}")
            if meta.get('thinker'):
                lines.append(f"    Thinker: {meta['thinker']}")
            lines.append(f"    {chunk['text'][:800]}...")  # Truncate long chunks
            lines.append("")
        
        return "\n".join(lines)


def main():
    """CLI entry point for reindexing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline for Curriculum KB")
    parser.add_argument("--reindex", action="store_true", help="Reindex knowledge base")
    parser.add_argument("--query", type=str, help="Test query")
    parser.add_argument("--subject", type=str, help="Subject filter (history/geography)")
    
    args = parser.parse_args()
    
    pipeline = RAGPipeline()
    
    if args.reindex:
        print("Starting knowledge base reindexing...")
        result = pipeline.index_knowledge_base()
        print(f"\n✅ Indexed {result['chunks_indexed']} chunks from {result['files_indexed']} files")
        
    elif args.query:
        print(f"Query: {args.query}")
        results = pipeline.query(args.query, subject=args.subject)
        print(f"\nTop {len(results)} results:")
        for r in results:
            print(f"\n[{r['distance']:.3f}] {r['metadata'].get('heading', 'Unknown')}")
            print(f"    {r['text'][:200]}...")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
