from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from PIL import Image
import os

class TaskRelevanceResponse(BaseModel):
    not_relevant_objects: list[str] = Field(description="List of objects in the image that are not relevant for completing the task")
    not_relevant_backgrounds: list[str] = Field(description="List of backgrounds in the image that are not relevant for completing the task")


def gemini_flash(im, language_instruction):
    """Few-shot VLM call using Gemini 2.5 Flash to determine task-irrelevant objects."""
    # Initialize the Gemini client (expects GEMINI_API_KEY environment variable)
    client = genai.Client()

    # Load images directly with PIL (no manual base64 encoding required)
    path_to_img = "example"
    fewshot_img1 = Image.open(f"{path_to_img}/open the drawer.png")
    TASK1 = "open the drawer"
    fewshot_img2 = Image.open(f"{path_to_img}/put carrot on plate.png")
    TASK2 = "put carrot on plate"
    fewshot_img3 = Image.open(f"{path_to_img}/put the eggplant in the blue sink.png")
    TASK3 = "put the eggplant in the blue sink"
    fewshot_img4 = Image.open(f"{path_to_img}/put the spoon on the towel.png")
    TASK4 = "put the spoon on the towel"
    fewshot_img5 = Image.open(f"{path_to_img}/stack the green block on the yellow block.png")
    TASK5 = "stack the green block on the yellow block"

    test_img = im

    contents = [
        "You will be shown some text and images.",
        # Example 1
        "Example 1",
        "Task: TASK1",
        fewshot_img1,
        '["obj1", "obj2"]',
        '["background1", "background2", "background3", "background4"]',
        # Example 2
        "Example 2",
        "Task: TASK2",
        fewshot_img2,
        '["obj1", "obj2", "obj3"]',
        '["background1", "background2", "background3"]',
        # Example 3
        "Example 3",
        "Task: TASK3",
        fewshot_img3,
        '["obj1", "obj2"]',
        '["background1", "background2", "background3", "background4"]',
        # Example 4
        "Example 4",
        "Task: TASK4",
        fewshot_img4,
        '["obj1", "obj2", "obj3"]',
        '["background1", "background2", "background3", "background4"]',
        # Example 5
        "Example 5",
        "Task: TASK5",
        fewshot_img5,
        '["obj1", "obj2", "obj3"]',
        '["background1", "background2"]',
        # Query
        "The robotic arm in the image is given the following task: "
        + language_instruction
        + ". Provide a list of objects in the image that are not relevant for completing the task, called 'not_relevant_objects'. Then provide a list of backgrounds in the image that are not relevant for completing the task, called 'not_relevant_backgrounds'. Give your answer in the form of two different lists with one or two words per object. Respond in JSON file format only.",
        test_img,
    ]

    # Call Gemini Flash with System Instructions and Enforced JSON Schema
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an assistant helping a robot determine what objects "
                "in the image are relevant for completing its task."
            ),
            response_mime_type="application/json",
            response_schema=TaskRelevanceResponse,
            max_output_tokens=2000,
            temperature=0.7,
        ),
    )

    # Automatically parsed into structured Python objects/JSON
    return response.text

if __name__ == "__main__":
    # Define directories
    widowx_path = "widowx"
    google_path = "google"

    dataset_dirs = [
        (widowx_path, "output/json/widowx"),
        (google_path, "output/json/google"),
    ]

    for source_dir, save_dir in dataset_dirs:
        if not os.path.exists(source_dir):
            continue

        os.makedirs(save_dir, exist_ok=True)

        for filename in os.listdir(source_dir):
            if filename.lower().endswith(".png"):
                file_path = os.path.join(source_dir, filename)
                language_instruction = os.path.splitext(filename)[0]

                img_pil = Image.open(file_path)
                response_json = gemini_flash(img_pil, language_instruction)
                print(f"Task: {language_instruction}")
                # Save output to JSON file
                output_json_path = os.path.join(save_dir, f"{language_instruction}.json")
                with open(output_json_path, "w", encoding="utf-8") as f:
                    f.write(response_json)
                print(f"Saved: {output_json_path}")
