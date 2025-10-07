#!/usr/bin/env python3

import sys
import numpy as np
from pathlib import Path
from rag.indexing import MultiModalIndex
from rag.models import ClipMultiModal
from rag.retrieval import Retriever
from typing import List, Dict, Any, Tuple
import re

class SimpleAccuracyImprover:
    def __init__(self):
        print("🚀 Initializing Simple Accuracy Improvement System...")
        self.clip = ClipMultiModal()
        self.index_dir = Path("index")
        
        if self.index_dir.exists():
            self.index = MultiModalIndex(persist_dir=self.index_dir, persistent=True)
        else:
            raise ValueError("No index found!")
        
        self.retriever = Retriever(index=self.index, clip=self.clip)
        
        # Domain-specific keywords for boosting
        self.domain_keywords = {
            'code': ['algorithm', 'function', 'code', 'programming', 'include', 'printf', 'int', 'main', 'binary', 'search', 'c++', 'python', 'java'],
            'email': ['email', 'gmail', 'inbox', 'compose', 'mail', 'message', 'send', 'receive', 'attachment'],
            'ui': ['interface', 'button', 'window', 'menu', 'dialog', 'screen', 'app', 'application', 'user']
        }

    def generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations to improve matching"""
        variations = [query]  # Original query
        
        # Synonyms and related terms
        synonyms = {
            'code': ['programming', 'algorithm', 'function', 'script', 'program', 'software'],
            'email': ['mail', 'message', 'gmail', 'inbox', 'envelope'],
            'ui': ['interface', 'screen', 'application', 'app', 'gui'],
            'screenshot': ['image', 'picture', 'screen capture', 'screen shot']
        }
        
        query_lower = query.lower()
        
        # Add synonym variations
        for key, values in synonyms.items():
            if key in query_lower:
                for synonym in values:
                    variation = query_lower.replace(key, synonym)
                    variations.append(variation)
        
        # Add specific domain terms
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    variations.append(f"{query} {domain}")
                    break
        
        return list(set(variations))  # Remove duplicates

    def enhanced_search_with_boosting(self, query: str, k: int = 5) -> List[Any]:
        """Enhanced search with domain boosting and query variations"""
        all_results = {}  # doc_id -> best_result
        
        # 1. Original query
        original_results = self.retriever.search_text(query, k=k*2)
        self._add_results_to_collection(all_results, original_results, "original")
        
        # 2. Query variations
        variations = self.generate_query_variations(query)
        for variation in variations[:3]:  # Use top 3 variations
            if variation != query:  # Skip original
                var_results = self.retriever.search_text(variation, k=k)
                self._add_results_to_collection(all_results, var_results, "variation")
        
        # 3. Apply domain-specific boosting
        query_lower = query.lower()
        detected_domain = self._detect_domain(query_lower)
        
        if detected_domain:
            for doc_id, result in all_results.items():
                boost = self._calculate_domain_boost(result, detected_domain)
                result.boosted_score = result.score * boost
        else:
            for result in all_results.values():
                result.boosted_score = result.score
        
        # 4. Sort by boosted score and return top k
        sorted_results = sorted(all_results.values(), key=lambda x: x.boosted_score, reverse=True)
        return sorted_results[:k]

    def _add_results_to_collection(self, collection: Dict, results: List, search_type: str):
        """Add results to collection, keeping the best score for each document"""
        for result in results:
            doc_id = result.metadata.get('source_name', str(hash(str(result))))
            
            if doc_id in collection:
                # Keep the better scoring result
                if result.score > collection[doc_id].score:
                    collection[doc_id] = result
            else:
                collection[doc_id] = result
            
            # Mark search type
            collection[doc_id].search_type = search_type

    def _detect_domain(self, query_lower: str) -> str:
        """Detect the domain of the query"""
        for domain, keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        return None

    def _calculate_domain_boost(self, result, domain: str) -> float:
        """Calculate domain-specific boost factor"""
        if not domain:
            return 1.0
        
        ocr_text = result.metadata.get('ocr_text', '').lower()
        
        # Count domain keyword matches
        keyword_matches = 0
        for keyword in self.domain_keywords[domain]:
            if keyword in ocr_text:
                keyword_matches += 1
        
        # Apply boost: 1.0 + (0.15 per keyword match)
        boost_factor = 1.0 + (keyword_matches * 0.15)
        return min(boost_factor, 2.0)  # Cap at 2.0x boost

    def enhanced_keyword_matching(self, query: str, results: List) -> List:
        """Enhanced keyword matching within results"""
        query_words = set(query.lower().split())
        
        for result in results:
            ocr_text = result.metadata.get('ocr_text', '').lower()
            ocr_words = set(ocr_text.split())
            
            # Calculate keyword overlap
            overlap = len(query_words.intersection(ocr_words))
            total_query_words = len(query_words)
            
            if total_query_words > 0:
                keyword_score = overlap / total_query_words
                # Combine with original score
                result.final_score = (result.boosted_score * 0.7) + (keyword_score * 0.3)
            else:
                result.final_score = result.boosted_score
        
        return sorted(results, key=lambda x: x.final_score, reverse=True)

    def comprehensive_accuracy_test(self):
        """Test the improved system"""
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
            
            # Enhanced method
            enhanced_results = self.enhanced_search_with_boosting(test['query'], k=3)
            
            # Apply final keyword matching
            if enhanced_results:
                enhanced_results = self.enhanced_keyword_matching(test['query'], enhanced_results)
            
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
            
            # Show top result details
            if enhanced_results:
                top_result = enhanced_results[0]
                boost_factor = getattr(top_result, 'boosted_score', 0) / top_result.score if top_result.score != 0 else 1
                print(f"   Boost Factor: {boost_factor:.2f}x")
                print(f"   Search Type: {getattr(top_result, 'search_type', 'unknown')}")
        
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
        
        original_accuracy = original_correct / total_tests if total_tests > 0 else 0
        enhanced_accuracy = enhanced_correct / total_tests if total_tests > 0 else 0
        improvement_rate = improvements / total_tests if total_tests > 0 else 0
        
        return {
            'original_accuracy': original_accuracy,
            'enhanced_accuracy': enhanced_accuracy,
            'absolute_improvement': enhanced_accuracy - original_accuracy,
            'improvement_rate': improvement_rate,
            'tests_improved': improvements,
            'total_tests': total_tests
        }

    def analyze_problematic_queries(self, comparisons):
        """Analyze queries that still fail"""
        print("\n🔬 DETAILED FAILURE ANALYSIS")
        print("-" * 60)
        
        failed_queries = [c for c in comparisons if not c['enhanced_relevant']]
        
        for failed in failed_queries:
            print(f"\n❌ FAILED: '{failed['query']}'")
            print(f"   Expected: {failed['expected']}")
            print(f"   Category: {failed['category']}")
            print(f"   Enhanced Score: {failed['enhanced_score']:.4f}")
            
            # Get top result details for diagnosis
            results = self.enhanced_search_with_boosting(failed['query'], k=1)
            if results:
                top_result = results[0]
                ocr_text = top_result.metadata.get('ocr_text', '')[:100]
                print(f"   Got: {top_result.metadata.get('type', 'unknown')}")
                print(f"   Source: {top_result.metadata.get('source_name', 'unknown')}")
                print(f"   OCR Preview: {ocr_text}...")
            
            # Suggest improvements
            self._suggest_improvements_for_query(failed)

    def _suggest_improvements_for_query(self, failed_query):
        """Suggest specific improvements for a failed query"""
        suggestions = []
        
        query_lower = failed_query['query'].lower()
        expected = failed_query['expected']
        
        if expected == 'email' and 'email' not in query_lower and 'gmail' not in query_lower:
            suggestions.append("Add explicit email keywords to query")
        
        if expected == 'code' and 'code' not in query_lower and 'programming' not in query_lower:
            suggestions.append("Add explicit programming keywords to query")
        
        if expected == 'ui' and 'interface' not in query_lower and 'app' not in query_lower:
            suggestions.append("Add explicit UI/interface keywords to query")
        
        if suggestions:
            print(f"   Suggestions: {'; '.join(suggestions)}")

def main():
    try:
        improver = SimpleAccuracyImprover()
        
        # Run comprehensive accuracy test
        print("🚀 Starting Simple Accuracy Improvement Test...")
        comparisons = improver.comprehensive_accuracy_test()
        
        # Calculate metrics
        metrics = improver.calculate_improvement_metrics(comparisons)
        
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
            print(f"\n⚠️ Limited improvement detected. Analyzing problematic queries...")
            improver.analyze_problematic_queries(comparisons)
        
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
            original_acc = data['original'] / data['total'] if data['total'] > 0 else 0
            enhanced_acc = data['enhanced'] / data['total'] if data['total'] > 0 else 0
            improvement = enhanced_acc - original_acc
            
            status = "📈" if improvement > 0 else "➡️"
            print(f"{status} {category}: {original_acc:.1%} → {enhanced_acc:.1%} (+{improvement:.1%})")
            
    except Exception as e:
        print(f"❌ Error in accuracy improvement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()