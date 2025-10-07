#!/usr/bin/env python3

import sys
import numpy as np
from pathlib import Path
from rag.indexing import MultiModalIndex
from rag.models import ClipMultiModal
from rag.retrieval import Retriever
from sentence_transformers import SentenceTransformer
import easyocr
import pytesseract
from PIL import Image
import json
import re
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AdvancedAccuracySystem:
    def __init__(self):
        print("🚀 Initializing Advanced Accuracy System...")
        self.clip = ClipMultiModal()
        self.index_dir = Path("index")
        
        if self.index_dir.exists():
            self.index = MultiModalIndex(persist_dir=self.index_dir, persistent=True)
        else:
            raise ValueError("No index found!")
        
        self.retriever = Retriever(index=self.index, clip=self.clip)
        
        # Initialize advanced components
        self.init_advanced_components()

    def init_advanced_components(self):
        """Initialize advanced embedding models and OCR engines"""
        print("📚 Loading advanced embedding models...")
        
        # Multiple embedding models for ensemble
        try:
            self.text_embedding_models = {
                'minilm': SentenceTransformer('all-MiniLM-L6-v2'),
                'mpnet': SentenceTransformer('all-mpnet-base-v2'),  
                'distilbert': SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
            }
            print("✅ Loaded multiple text embedding models")
        except Exception as e:
            print(f"⚠️ Could not load all embedding models: {e}")
            self.text_embedding_models = {}
        
        # Multiple OCR engines  
        try:
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            print("✅ Loaded EasyOCR engine")
        except Exception as e:
            print(f"⚠️ Could not load EasyOCR: {e}")
            self.ocr_reader = None
        
        # Keyword extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Domain-specific keywords
        self.domain_keywords = {
            'code': ['algorithm', 'function', 'code', 'programming', 'include', 'printf', 'int', 'main', 'binary', 'search', 'c++', 'python', 'java'],
            'email': ['email', 'gmail', 'inbox', 'compose', 'mail', 'message', 'send', 'receive', 'attachment'],
            'ui': ['interface', 'button', 'window', 'menu', 'dialog', 'screen', 'app', 'application', 'user']
        }

    def enhanced_ocr_extraction(self, image_path: str) -> Dict[str, Any]:
        """Extract text using multiple OCR engines and combine results"""
        results = {'combined_text': '', 'confidence': 0.0, 'methods': []}
        
        try:
            image = Image.open(image_path)
            
            # Method 1: EasyOCR
            if self.ocr_reader:
                try:
                    easyocr_results = self.ocr_reader.readtext(str(image_path))
                    easyocr_text = ' '.join([result[1] for result in easyocr_results])
                    easyocr_confidence = np.mean([result[2] for result in easyocr_results]) if easyocr_results else 0
                    results['methods'].append({
                        'method': 'EasyOCR',
                        'text': easyocr_text,
                        'confidence': easyocr_confidence
                    })
                except Exception as e:
                    print(f"EasyOCR failed: {e}")
            
            # Method 2: Tesseract
            try:
                tesseract_text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
                tesseract_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in tesseract_data['conf'] if int(conf) > 0]
                tesseract_confidence = np.mean(confidences) / 100.0 if confidences else 0
                
                results['methods'].append({
                    'method': 'Tesseract',
                    'text': tesseract_text,
                    'confidence': tesseract_confidence
                })
            except Exception as e:
                print(f"Tesseract failed: {e}")
            
            # Combine results (weighted by confidence)
            if results['methods']:
                total_weighted_text = []
                total_weight = 0
                
                for method in results['methods']:
                    weight = method['confidence']
                    total_weighted_text.append(method['text'])
                    total_weight += weight
                
                # Use the text from the most confident method, but include all for fallback
                best_method = max(results['methods'], key=lambda x: x['confidence'])
                results['combined_text'] = best_method['text']
                results['confidence'] = best_method['confidence']
                
                # Also keep all text combined for broader matching
                results['all_text_combined'] = ' '.join([m['text'] for m in results['methods']])
        
        except Exception as e:
            print(f"Enhanced OCR extraction failed: {e}")
        
        return results

    def generate_query_expansions(self, query: str) -> List[str]:
        """Generate expanded and reformulated queries"""
        expansions = [query]  # Original query
        
        # Synonyms and related terms
        synonyms = {
            'code': ['programming', 'algorithm', 'function', 'script', 'program'],
            'email': ['mail', 'message', 'gmail', 'inbox'],
            'ui': ['interface', 'screen', 'application', 'app'],
            'screenshot': ['image', 'picture', 'screen capture']
        }
        
        query_lower = query.lower()
        for key, values in synonyms.items():
            if key in query_lower:
                for synonym in values:
                    expanded = query_lower.replace(key, synonym)
                    expansions.append(expanded)
        
        # Add domain-specific expansions
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    expansions.append(f"{query} {domain}")
                    break
        
        return list(set(expansions))  # Remove duplicates

    def hybrid_search(self, query: str, k: int = 5) -> List[Any]:
        """Implement hybrid search combining semantic and keyword matching"""
        results_sets = []
        
        # 1. Original CLIP-based semantic search
        semantic_results = self.retriever.search_text(query, k=k*2)
        if semantic_results:
            for result in semantic_results:
                result.search_method = 'semantic'
                result.original_score = result.score
            results_sets.append(semantic_results)
        
        # 2. Keyword-based search using TF-IDF
        keyword_results = self.keyword_search(query, k=k*2)
        if keyword_results:
            results_sets.append(keyword_results)
        
        # 3. Query expansion results
        query_expansions = self.generate_query_expansions(query)
        for expanded_query in query_expansions[1:3]:  # Use top 2 expansions
            expanded_results = self.retriever.search_text(expanded_query, k=k)
            if expanded_results:
                for result in expanded_results:
                    result.search_method = 'expansion'
                    result.expanded_query = expanded_query
                results_sets.append(expanded_results)
        
        # 4. Domain-aware boosting
        domain_results = self.domain_aware_search(query, k=k)
        if domain_results:
            results_sets.append(domain_results)
        
        # Combine and re-rank results
        return self.combine_and_rerank_results(query, results_sets, k)

    def keyword_search(self, query: str, k: int = 5) -> List[Any]:
        """TF-IDF based keyword search"""
        try:
            # Get all documents from index
            all_docs = []
            doc_metadata = []
            
            for doc_id in self.index.ids:
                doc = self.index.get_document(doc_id)
                if doc and hasattr(doc, 'metadata'):
                    ocr_text = doc.metadata.get('ocr_text', '')
                    all_docs.append(ocr_text)
                    doc_metadata.append((doc, doc_id))
            
            if not all_docs:
                return []
            
            # Fit TF-IDF and search
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_docs)
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum threshold
                    doc, doc_id = doc_metadata[idx]
                    doc.score = similarities[idx]
                    doc.search_method = 'keyword'
                    results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"Keyword search failed: {e}")
            return []

    def domain_aware_search(self, query: str, k: int = 5) -> List[Any]:
        """Search with domain-specific boosting"""
        # Detect query domain
        query_lower = query.lower()
        detected_domain = None
        
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_domain = domain
                break
        
        if not detected_domain:
            return []
        
        # Get semantic results and boost domain-relevant ones
        results = self.retriever.search_text(query, k=k*2)
        
        for result in results:
            ocr_text = result.metadata.get('ocr_text', '').lower()
            
            # Calculate domain relevance boost
            domain_score = 0
            for keyword in self.domain_keywords[detected_domain]:
                if keyword in ocr_text:
                    domain_score += 1
            
            # Apply boost
            if domain_score > 0:
                boost_factor = 1.0 + (domain_score * 0.2)  # 20% boost per keyword
                result.score = result.score * boost_factor
                result.search_method = 'domain_boosted'
                result.domain_boost = boost_factor
        
        # Re-sort by boosted scores
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def combine_and_rerank_results(self, query: str, results_sets: List[List], k: int = 5) -> List[Any]:
        """Combine multiple result sets and re-rank using ensemble scoring"""
        # Collect all unique results
        all_results = {}  # doc_id -> result
        
        for results in results_sets:
            for result in results:
                doc_id = result.metadata.get('id', str(hash(str(result))))
                
                if doc_id in all_results:
                    # Combine scores from multiple methods
                    existing = all_results[doc_id]
                    existing.combined_score = getattr(existing, 'combined_score', existing.score) + result.score
                    existing.search_methods = getattr(existing, 'search_methods', [existing.search_method]) + [result.search_method]
                else:
                    result.combined_score = result.score
                    result.search_methods = [result.search_method]
                    all_results[doc_id] = result
        
        # Normalize combined scores
        combined_results = list(all_results.values())
        if not combined_results:
            return []
        
        max_combined = max(r.combined_score for r in combined_results)
        if max_combined > 0:
            for result in combined_results:
                result.normalized_score = result.combined_score / max_combined
        
        # Final ranking with method diversity bonus
        for result in combined_results:
            method_diversity_bonus = len(set(result.search_methods)) * 0.1
            result.final_score = result.normalized_score + method_diversity_bonus
        
        # Sort by final score and return top k
        combined_results.sort(key=lambda x: x.final_score, reverse=True)
        return combined_results[:k]

    def comprehensive_accuracy_test(self):
        """Test the improved system against the original problems"""
        print("\n🧪 COMPREHENSIVE ACCURACY TEST")
        print("=" * 60)
        
        test_cases = [
            {"query": "email screenshot", "expected": "email", "category": "Email"},
            {"query": "binary search algorithm", "expected": "code", "category": "Code"},
            {"query": "programming code", "expected": "code", "category": "Code"},
            {"query": "user interface", "expected": "ui", "category": "UI"},
            {"query": "gmail inbox", "expected": "email", "category": "Email"},
            {"query": "c programming", "expected": "code", "category": "Code"},
            {"query": "algorithm implementation", "expected": "code", "category": "Code"},
            {"query": "screenshot interface", "expected": "ui", "category": "UI"},
            {"query": "data structures", "expected": "code", "category": "Code"},
            {"query": "application window", "expected": "ui", "category": "UI"},
        ]
        
        results_comparison = []
        
        for test in test_cases:
            print(f"\n🔍 Testing: '{test['query']}'")
            
            # Original method
            original_results = self.retriever.search_text(test['query'], k=3)
            
            # Enhanced hybrid method
            enhanced_results = self.hybrid_search(test['query'], k=3)
            
            # Compare results
            original_relevant = self._is_result_relevant(original_results[0] if original_results else None, test['expected'])
            enhanced_relevant = self._is_result_relevant(enhanced_results[0] if enhanced_results else None, test['expected'])
            
            comparison = {
                'query': test['query'],
                'expected': test['expected'],
                'category': test['category'],
                'original_relevant': original_relevant,
                'enhanced_relevant': enhanced_relevant,
                'original_score': original_results[0].score if original_results else 0,
                'enhanced_score': enhanced_results[0].final_score if enhanced_results else 0,
                'improvement': enhanced_relevant and not original_relevant
            }
            
            results_comparison.append(comparison)
            
            # Show detailed results
            status_original = "✅" if original_relevant else "❌"
            status_enhanced = "✅" if enhanced_relevant else "❌"
            improvement_icon = "📈" if comparison['improvement'] else "➡️"
            
            print(f"   Original: {status_original} Score: {comparison['original_score']:.4f}")
            print(f"   Enhanced: {status_enhanced} Score: {comparison['enhanced_score']:.4f}")
            print(f"   {improvement_icon} {'IMPROVED!' if comparison['improvement'] else 'Same result'}")
        
        return results_comparison

    def _is_result_relevant(self, result, expected_type):
        """Check if result matches expected type"""
        if not result:
            return False
            
        ocr_text = result.metadata.get('ocr_text', '').lower()
        
        if expected_type == "email":
            return any(term in ocr_text for term in ['email', 'gmail', 'inbox', 'compose', 'mail'])
        elif expected_type == "code":
            return any(term in ocr_text for term in ['code', 'include', 'printf', 'int', 'main', 'algorithm'])
        elif expected_type == "ui":
            return any(term in ocr_text for term in ['interface', 'button', 'window', 'menu', 'dialog'])
        
        return False

    def calculate_improvement_metrics(self, comparisons):
        """Calculate improvement metrics"""
        total_tests = len(comparisons)
        original_correct = sum(1 for c in comparisons if c['original_relevant'])
        enhanced_correct = sum(1 for c in comparisons if c['enhanced_relevant'])
        improvements = sum(1 for c in comparisons if c['improvement'])
        
        original_accuracy = original_correct / total_tests
        enhanced_accuracy = enhanced_correct / total_tests
        improvement_rate = improvements / total_tests
        
        return {
            'original_accuracy': original_accuracy,
            'enhanced_accuracy': enhanced_accuracy,
            'absolute_improvement': enhanced_accuracy - original_accuracy,
            'improvement_rate': improvement_rate,
            'tests_improved': improvements,
            'total_tests': total_tests
        }

def main():
    try:
        system = AdvancedAccuracySystem()
        
        # Run comprehensive accuracy test
        print("\n🚀 Starting Advanced Accuracy Improvement Test...")
        comparisons = system.comprehensive_accuracy_test()
        
        # Calculate metrics
        metrics = system.calculate_improvement_metrics(comparisons)
        
        print("\n📊 IMPROVEMENT RESULTS")
        print("=" * 50)
        print(f"Original Accuracy: {metrics['original_accuracy']:.1%}")
        print(f"Enhanced Accuracy: {metrics['enhanced_accuracy']:.1%}")
        print(f"Absolute Improvement: +{metrics['absolute_improvement']:.1%}")
        print(f"Tests Improved: {metrics['tests_improved']}/{metrics['total_tests']}")
        print(f"Improvement Rate: {metrics['improvement_rate']:.1%}")
        
        if metrics['absolute_improvement'] > 0:
            print(f"\n🎉 SUCCESS: System accuracy improved by {metrics['absolute_improvement']:.1%}!")
        else:
            print(f"\n⚠️ No significant improvement detected. Further optimization needed.")
        
        # Show category breakdown
        print(f"\n📈 CATEGORY BREAKDOWN")
        categories = {}
        for comp in comparisons:
            cat = comp['category']
            if cat not in categories:
                categories[cat] = {'original': 0, 'enhanced': 0, 'total': 0}
            categories[cat]['total'] += 1
            if comp['original_relevant']:
                categories[cat]['original'] += 1
            if comp['enhanced_relevant']:
                categories[cat]['enhanced'] += 1
        
        for category, data in categories.items():
            original_acc = data['original'] / data['total']
            enhanced_acc = data['enhanced'] / data['total']
            improvement = enhanced_acc - original_acc
            
            status = "📈" if improvement > 0 else "➡️"
            print(f"{status} {category}: {original_acc:.1%} → {enhanced_acc:.1%} (+{improvement:.1%})")
            
    except Exception as e:
        print(f"❌ Error in accuracy improvement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()