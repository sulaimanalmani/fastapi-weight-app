import os
import re
import io
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

app = FastAPI()

def compress_image(image_data, max_size=(1600, 1600), quality=100):
    """Compress the image by resizing and reducing quality."""
    with Image.open(io.BytesIO(image_data)) as img:
        img.thumbnail(max_size)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG", quality=quality)
        img_bytes.seek(0)
        return img_bytes.read()

@app.post("/extract-weight")
async def extract_weight(file: UploadFile = File(...)):
    try:
        # Read the file into memory
        image_data = await file.read()

        # Compress the image
        compressed_bytes = compress_image(image_data)

        # Optionally, if you have a service account JSON file, do:
        # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/app/service_account_key.json'
        # Otherwise, if you’re using Cloud Run with Workload Identity, you can skip that.

        # Initialize the GenAI client
        client = genai.Client(
            vertexai=True,
            project="my-fastapi-weight-extraction",  # Replace with your project
            location="us-central1"
        )

        # Prepare the image as a Part
        image_part = types.Part.from_bytes(
            data=compressed_bytes,
            mime_type="image/jpeg"
        )

        # Create the user request content
        contents = [
            types.Content(
                role="user",
                parts=[
                    image_part,
                    types.Part.from_text("Extract my weight from this image.")
                ]
            )
        ]

        # Define the generation config
        generate_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=8192,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
            ],
            response_mime_type="application/json",
        )

        weight_found = None
        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash-exp",
            contents=contents,
            config=generate_config
        ):
            raw_content = str(chunk)
            match = re.search(r'"weight":\s*"([\d.]+)"', raw_content)
            if match:
                weight_found = match.group(1)
                break

        if weight_found:
            return JSONResponse({"weight": weight_found})
        else:
            return JSONResponse({"message": "Weight not found."})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
