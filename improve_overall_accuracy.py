#!/usr/bin/env python3

import sys
import numpy as np
from pathlib import Path
from rag.indexing import MultiModalIndex
from rag.models import ClipMultiModal
from rag.retrieval import Retriever
from typing import List, Dict, Any
import re

class AccuracyImprover:
    def __init__(self):
        self.clip = ClipMultiModal()
        self.index_dir = Path("index")
        if self.index_dir.exists():
            self.index = MultiModalIndex(persist_dir=self.index_dir, persistent=True)
        else:
            raise ValueError("No index found!")
        self.retriever = Retriever(index=self.index, clip=self.clip)

    def analyze_current_problems(self):
        """Comprehensive analysis of accuracy problems"""
        print("🔍 COMPREHENSIVE ACCURACY ANALYSIS")
        print("=" * 60)
        
        # Test diverse queries
        test_cases = [
            {"query": "email screenshot", "expected": "email", "category": "UI"},
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
        
        problems = []
        
        for test in test_cases:
            results = self.retriever.search_text(test["query"], k=5)
            if results:
                top_result = results[0]
                problems.append({
                    "query": test["query"],
                    "expected": test["expected"],
                    "category": test["category"],
                    "top_score": top_result.score,
                    "top_type": top_result.metadata.get('type', 'unknown'),
                    "top_source": top_result.metadata.get('source_name', 'unknown'),
                    "is_relevant": self._is_result_relevant(top_result, test["expected"])
                })
        
        return problems

    def _is_result_relevant(self, result, expected_type):
        """Check if result matches expected type"""
        ocr_text = result.metadata.get('ocr_text', '').lower()
        
        if expected_type == "email":
            return any(term in ocr_text for term in ['email', 'gmail', 'inbox', 'compose', 'mail'])
        elif expected_type == "code":
            return any(term in ocr_text for term in ['code', 'include', 'printf', 'int', 'main', 'algorithm'])
        elif expected_type == "ui":
            return any(term in ocr_text for term in ['interface', 'button', 'window', 'menu', 'dialog'])
        
        return False

    def calculate_accuracy_metrics(self, problems):
        """Calculate comprehensive accuracy metrics"""
        total_queries = len(problems)
        correct_top1 = sum(1 for p in problems if p["is_relevant"])
        
        # Score distribution
        scores = [p["top_score"] for p in problems]
        avg_score = np.mean(scores)
        score_variance = np.var(scores)
        
        # Category performance
        categories = {}
        for p in problems:
            cat = p["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "correct": 0}
            categories[cat]["total"] += 1
            if p["is_relevant"]:
                categories[cat]["correct"] += 1
        
        return {
            "overall_accuracy": correct_top1 / total_queries,
            "avg_score": avg_score,
            "score_variance": score_variance,
            "category_performance": {
                cat: data["correct"] / data["total"] 
                for cat, data in categories.items()
            },
            "problems": problems
        }

    def identify_root_causes(self, metrics):
        """Identify specific root causes of accuracy problems"""
        causes = []
        
        # Low overall accuracy
        if metrics["overall_accuracy"] < 0.7:
            causes.append({
                "issue": "Low Overall Accuracy",
                "severity": "HIGH",
                "description": f"Only {metrics['overall_accuracy']:.1%} of queries return relevant results",
                "impact": "Users get wrong content for most searches"
            })
        
        # Poor score distribution
        if metrics["avg_score"] < 0.1:
            causes.append({
                "issue": "Low Similarity Scores", 
                "severity": "HIGH",
                "description": f"Average similarity score is {metrics['avg_score']:.3f}",
                "impact": "Embeddings don't capture semantic similarity well"
            })
        
        # High variance in scores
        if metrics["score_variance"] > 0.1:
            causes.append({
                "issue": "Inconsistent Scoring",
                "severity": "MEDIUM", 
                "description": f"Score variance is {metrics['score_variance']:.3f}",
                "impact": "Unpredictable ranking quality"
            })
        
        # Category-specific issues
        for category, accuracy in metrics["category_performance"].items():
            if accuracy < 0.5:
                causes.append({
                    "issue": f"Poor {category} Recognition",
                    "severity": "MEDIUM",
                    "description": f"{category} queries only {accuracy:.1%} accurate",
                    "impact": f"Users can't find {category.lower()} content reliably"
                })
        
        return causes

    def generate_solutions(self, causes):
        """Generate specific solutions for identified problems"""
        solutions = {}
        
        for cause in causes:
            issue = cause["issue"]
            
            if "Low Overall Accuracy" in issue:
                solutions["embedding_model"] = {
                    "priority": "HIGH",
                    "actions": [
                        "Switch to better embedding model (e.g., all-MiniLM-L6-v2)",
                        "Use domain-specific embeddings for technical content", 
                        "Implement hybrid search (semantic + keyword)",
                        "Add query expansion and reformulation"
                    ]
                }
            
            if "Low Similarity Scores" in issue:
                solutions["scoring_system"] = {
                    "priority": "HIGH", 
                    "actions": [
                        "Normalize similarity scores to 0-1 range",
                        "Use different similarity metrics (dot product vs cosine)",
                        "Add learned relevance scoring",
                        "Implement query-document matching features"
                    ]
                }
            
            if "Inconsistent Scoring" in issue:
                solutions["score_calibration"] = {
                    "priority": "MEDIUM",
                    "actions": [
                        "Implement score normalization across modalities",
                        "Add confidence intervals to scores",
                        "Use ensemble scoring methods", 
                        "Calibrate scores based on query types"
                    ]
                }
            
            if "Recognition" in issue:
                solutions["content_processing"] = {
                    "priority": "MEDIUM",
                    "actions": [
                        "Improve OCR preprocessing and post-processing",
                        "Add multiple OCR engines (EasyOCR + Tesseract)",
                        "Use vision-language models for image understanding",
                        "Implement content-specific keyword extraction"
                    ]
                }
        
        return solutions

def main():
    try:
        improver = AccuracyImprover()
        
        print("Analyzing current accuracy problems...")
        problems = improver.analyze_current_problems()
        
        print("\nCalculating accuracy metrics...")
        metrics = improver.calculate_accuracy_metrics(problems)
        
        print("\n📊 ACCURACY METRICS")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.1%}")
        print(f"Average Score: {metrics['avg_score']:.4f}")
        print(f"Score Variance: {metrics['score_variance']:.4f}")
        
        print(f"\n📈 CATEGORY PERFORMANCE")
        for category, accuracy in metrics['category_performance'].items():
            status = "✅" if accuracy > 0.7 else "⚠️" if accuracy > 0.3 else "❌"
            print(f"{status} {category}: {accuracy:.1%}")
        
        print(f"\n🔍 ROOT CAUSE ANALYSIS")
        causes = improver.identify_root_causes(metrics)
        for cause in causes:
            severity_icon = "🔴" if cause["severity"] == "HIGH" else "🟡"
            print(f"{severity_icon} {cause['issue']} ({cause['severity']})")
            print(f"   Description: {cause['description']}")
            print(f"   Impact: {cause['impact']}")
        
        print(f"\n🛠️ RECOMMENDED SOLUTIONS")
        solutions = improver.generate_solutions(causes)
        for solution_type, details in solutions.items():
            priority_icon = "🔥" if details["priority"] == "HIGH" else "⭐"
            print(f"{priority_icon} {solution_type.replace('_', ' ').title()} ({details['priority']} PRIORITY)")
            for i, action in enumerate(details["actions"], 1):
                print(f"   {i}. {action}")
        
        print(f"\n📋 DETAILED QUERY ANALYSIS")
        print("-" * 60)
        for problem in problems[:5]:  # Show top 5 problematic queries
            status = "✅" if problem["is_relevant"] else "❌"
            print(f"{status} Query: '{problem['query']}'")
            print(f"   Expected: {problem['expected']}, Got: {problem['top_type']}")
            print(f"   Score: {problem['top_score']:.4f}, Source: {problem['top_source']}")
            print()
            
    except Exception as e:
        print(f"❌ Error in accuracy analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()