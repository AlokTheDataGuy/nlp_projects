from typing import Dict, Any, List

def aggregate_responses(task_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate responses from different models into a coherent response.
    
    Args:
        task_results: Dictionary with results from all tasks
        
    Returns:
        Dictionary with the aggregated response
    """
    response = {
        "text": "",
        "has_image": False,
        "image": None,
        "error": False
    }
    
    # Check for errors in any task
    for task_name, result in task_results.items():
        if result.get("error", False):
            response["error"] = True
            response["text"] += f"Error in {task_name}: {result.get('text', 'Unknown error')}\n"
    
    # If there was an error, return early
    if response["error"]:
        return response
    
    # Process image analysis if available
    if "image_processing" in task_results:
        img_result = task_results["image_processing"]
        response["text"] += f"Image Analysis: {img_result['text']}\n\n"
    
    # Process text response
    if "text_processing" in task_results:
        text_result = task_results["text_processing"]
        # If we already have image analysis, add a separator
        if response["text"]:
            response["text"] += "Additional thoughts: "
        response["text"] += text_result["text"]
    
    # Add generated image if available
    if "image_generation" in task_results:
        img_gen = task_results["image_generation"]
        response["has_image"] = True
        response["image"] = img_gen["image"]
        response["image_prompt"] = img_gen["prompt"]
    
    return response
