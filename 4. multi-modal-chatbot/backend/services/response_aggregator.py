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

    # Check if we have image processing results
    has_image_processing = "image_processing" in task_results

    # If we have image processing, prioritize that response
    if has_image_processing:
        img_result = task_results["image_processing"]
        # Use the image processing result directly without adding "Image Analysis:" prefix
        response["text"] = img_result['text']
    # Otherwise, use the text processing result
    elif "text_processing" in task_results:
        text_result = task_results["text_processing"]
        response["text"] = text_result["text"]

    # No image generation in this version

    return response
