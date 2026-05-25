# backend/scripts/setup_qdrant_cloud.py

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

COLLECTION_NAME = "Health_QA_CoT"

def main():
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    print("Loaded URL:", QDRANT_URL)

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    print("Connected to Qdrant Cloud:", QDRANT_URL)

    # Create collection
    try:
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            ),
        )
        print(f"✔ Collection '{COLLECTION_NAME}' created successfully.")
    except Exception as e:
        print("❌ Error creating collection:", e)

    # Create payload index
    print("Creating payload index for 'domain'...")

    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="domain",
            field_schema="keyword"
        )
        print("✔ Payload index created.")
    except Exception as e:
        print("ℹ Index already exists or error:", e)

    print("✔ Setup complete.")

if __name__ == "__main__":
    main()