"""
Model Comparison Tool

Provides functionality to compare different LLM models for article generation.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("model_comparison")


def load_benchmark_results(file_path: str) -> Dict[str, Any]:
    """Load benchmark results from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Error loading benchmark results from {file_path}: {str(e)}")
        return {}


def analyze_benchmark_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze benchmark results and add qualitative metrics."""
    if not results:
        return {"error": "No benchmark results to analyze"}
    
    # Add qualitative analysis to each article
    for model_id, model_data in results.items():
        if "results" not in model_data:
            continue
        
        total_quality_score = 0
        total_structure_score = 0
        total_readability_grade = 0
        
        for article_result in model_data["results"]:
            if "article" not in article_result or not article_result["article"]:
                continue
            
            # Analyze the article text
            try:
                from text_analysis import analyze_text
                analysis = analyze_text(article_result["article"])
                article_result["analysis"] = analysis
                
                # Update totals
                total_quality_score += analysis["quality_score"]
                total_structure_score += analysis["structure_score"]
                total_readability_grade += analysis["readability_grade"]
            except ImportError:
                logger.warning("Text analysis module not available")
        
        # Calculate averages
        num_articles = len(model_data["results"])
        if num_articles > 0:
            model_data["average_quality_score"] = total_quality_score / num_articles
            model_data["average_structure_score"] = total_structure_score / num_articles
            model_data["average_readability_grade"] = total_readability_grade / num_articles
    
    return results


def compare_models(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare models based on benchmark results."""
    if not results:
        return {"error": "No benchmark results to compare"}
    
    # Extract model metrics
    models = []
    for model_id, model_data in results.items():
        if "error" in model_data:
            continue
        
        model_metrics = {
            "model": model_id,
            "average_time": model_data.get("average_time", 0),
            "average_tokens": model_data.get("average_tokens", 0),
            "success_rate": model_data.get("success_rate", 0),
            "average_quality_score": model_data.get("average_quality_score", 0),
            "average_structure_score": model_data.get("average_structure_score", 0),
            "average_readability_grade": model_data.get("average_readability_grade", 0),
        }
        models.append(model_metrics)
    
    # Calculate overall scores
    for model in models:
        # Normalize scores (higher is better)
        time_score = 100 - min(100, (model["average_time"] / 10) * 20)  # Lower time is better
        token_score = min(100, (model["average_tokens"] / 10))  # Higher token count is better
        success_score = model["success_rate"] * 100
        quality_score = model["average_quality_score"]
        
        # Calculate weighted overall score
        model["overall_score"] = (
            time_score * 0.2 +
            token_score * 0.1 +
            success_score * 0.3 +
            quality_score * 0.4
        )
    
    # Sort models by overall score
    sorted_models = sorted(models, key=lambda x: x.get("overall_score", 0), reverse=True)
    
    return {
        "models": sorted_models,
        "best_model": sorted_models[0]["model"] if sorted_models else None,
    }


def generate_comparison_report(results: Dict[str, Any], output_file: str) -> bool:
    """Generate a detailed comparison report."""
    if not results or "models" not in results:
        logger.error("No valid comparison results to generate report")
        return False
    
    try:
        # Create report content
        report = ["# LLM Model Comparison Report for Article Generation\n"]
        
        # Add summary
        report.append("## Summary\n")
        report.append(f"Best performing model: **{results['best_model']}**\n")
        report.append("\nModel rankings by overall performance:\n")
        
        for i, model in enumerate(results["models"]):
            report.append(f"{i+1}. **{model['model']}** - Overall score: {model['overall_score']:.2f}\n")
        
        # Add detailed metrics
        report.append("\n## Detailed Metrics\n")
        report.append("| Model | Overall Score | Generation Time | Tokens | Success Rate | Quality Score | Structure Score | Readability |\n")
        report.append("|-------|--------------|----------------|--------|--------------|---------------|----------------|------------|\n")
        
        for model in results["models"]:
            report.append(
                f"| {model['model']} | {model['overall_score']:.2f} | "
                f"{model['average_time']:.2f}s | {model['average_tokens']:.1f} | "
                f"{model['success_rate']*100:.1f}% | {model['average_quality_score']:.2f} | "
                f"{model['average_structure_score']:.2f} | {model['average_readability_grade']:.2f} |\n"
            )
        
        # Add recommendations
        report.append("\n## Recommendations\n")
        best_model = results["models"][0]
        report.append(f"Based on the benchmark results, **{best_model['model']}** is recommended for article generation because:\n\n")
        
        # Determine strengths of the best model
        strengths = []
        if best_model["average_time"] == min(m["average_time"] for m in results["models"]):
            strengths.append("- It has the fastest generation time")
        if best_model["average_tokens"] == max(m["average_tokens"] for m in results["models"]):
            strengths.append("- It produces the most content (highest token count)")
        if best_model["success_rate"] == max(m["success_rate"] for m in results["models"]):
            strengths.append("- It has the highest success rate")
        if best_model["average_quality_score"] == max(m["average_quality_score"] for m in results["models"]):
            strengths.append("- It produces the highest quality articles")
        if best_model["average_structure_score"] == max(m["average_structure_score"] for m in results["models"]):
            strengths.append("- It creates the best structured articles")
        
        # Add default strength if none found
        if not strengths:
            strengths.append("- It has the best overall performance across all metrics")
        
        report.extend(strengths)
        
        # Write report to file
        with open(output_file, 'w') as f:
            f.write("\n".join(report))
        
        logger.info(f"Comparison report generated and saved to {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating comparison report: {str(e)}")
        return False


def main():
    """Main entry point for the model comparison tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Model Comparison Tool")
    parser.add_argument("--input", required=True, help="Path to benchmark results JSON file")
    parser.add_argument("--report", default="comparison_report.md", help="Path to save comparison report")
    
    args = parser.parse_args()
    
    # Load benchmark results
    results = load_benchmark_results(args.input)
    if not results:
        logger.error(f"Failed to load benchmark results from {args.input}")
        return
    
    # Analyze benchmark results
    analyzed_results = analyze_benchmark_results(results)
    
    # Compare models
    comparison = compare_models(analyzed_results)
    
    # Generate report
    generate_comparison_report(comparison, args.report)
    
    print(f"Comparison complete. Report saved to {args.report}.")


if __name__ == "__main__":
    main()
