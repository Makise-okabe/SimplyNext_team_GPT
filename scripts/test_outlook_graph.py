import os
import sys
from io import BytesIO

import msal
import requests
from pypdf import PdfReader
from dotenv import load_dotenv



load_dotenv()

CLIENT_ID = os.getenv("MS_CLIENT_ID")
AUTHORITY = os.getenv(
    "MS_AUTHORITY",
    "https://login.microsoftonline.com/consumers",
)

SCOPES = [
    "User.Read",
    "Mail.Read",
]

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"


def get_access_token() -> str:
    if not CLIENT_ID:
        print("ERROR: MS_CLIENT_ID is missing from .env")
        sys.exit(1)

    app = msal.PublicClientApplication(
        client_id=CLIENT_ID,
        authority=AUTHORITY,
    )

    result = app.acquire_token_interactive(
        scopes=SCOPES
    )

    if "access_token" not in result:
        print("\nLOGIN FAILED")
        print(result)
        sys.exit(1)

    return result["access_token"]


def graph_get(endpoint: str, token: str, params=None):
    response = requests.get(
        f"{GRAPH_BASE_URL}{endpoint}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
        params=params,
        timeout=30,
    )

    if not response.ok:
        print("\nGRAPH REQUEST FAILED")
        print("Status:", response.status_code)
        print(response.text)
        sys.exit(1)

    return response.json()

def graph_get_bytes(endpoint: str, token: str) -> bytes:
    response = requests.get(
        f"{GRAPH_BASE_URL}{endpoint}",
        headers={
            "Authorization": f"Bearer {token}",
        },
        timeout=30,
    )

    if not response.ok:
        print("\nGRAPH BINARY REQUEST FAILED")
        print("Status:", response.status_code)
        print(response.text)
        sys.exit(1)

    return response.content

def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)

def get_attachments(message_id: str, token: str):
    return graph_get(
        f"/me/messages/{message_id}/attachments",
        token,
        params={
            "$select": "id,name,contentType,size,isInline",
        },
    )


def main():
    print("=" * 70)
    print("SimplyNext Outlook Graph Test")
    print("=" * 70)

    print("\nOpening Microsoft login...")
    token = get_access_token()

    # 1. Check which Microsoft account we actually logged into
    me = graph_get(
        "/me",
        token,
        params={
            "$select": "displayName,userPrincipalName,mail",
        },
    )

    account = me.get("mail") or me.get("userPrincipalName")

    print("\nCONNECTED ACCOUNT")
    print("-" * 70)
    print("Name :", me.get("displayName"))
    print("Email:", account)

    # 2. Read latest Inbox emails
    messages = graph_get(
        "/me/mailFolders/inbox/messages",
        token,
        params={
            "$top": "10",
            "$select": (
                "id,subject,from,receivedDateTime,"
                "hasAttachments,bodyPreview"
            ),
            "$orderby": "receivedDateTime desc",
        },
    )

    emails = messages.get("value", [])

    print("\nRECENT INBOX EMAILS")
    print("=" * 70)

    if not emails:
        print("Inbox is empty.")
        return

    for i, email in enumerate(emails, start=1):
        sender = (
            email.get("from", {})
            .get("emailAddress", {})
        )

        print(f"\nEMAIL {i}/{len(emails)}")
        print("-" * 70)
        print("Subject     :", email.get("subject"))
        print("Sender name :", sender.get("name"))
        print("Sender email:", sender.get("address"))
        print("Received    :", email.get("receivedDateTime"))
        print("Attachments :", email.get("hasAttachments"))
        print("Message ID  :", email.get("id"))

        preview = email.get("bodyPreview", "")
        if len(preview) > 300:
            preview = preview[:300] + "..."

        print("Preview     :", preview)
        
        # Test attachment retrieval using the McKinsey email
        subject = email.get("subject") or ""

        if email.get("hasAttachments") and "McKinsey" in subject:
            attachments = get_attachments(email["id"], token)
            attachment_list = attachments.get("value", [])

            print("\n    ATTACHMENTS")
            print("    " + "-" * 60)

            if not attachment_list:
                print("    Graph says hasAttachments=True, but no attachments returned.")

            for j, attachment in enumerate(attachment_list, start=1):
                print(f"    ATTACHMENT {j}")
                print("    Name      :", attachment.get("name"))
                print("    Type      :", attachment.get("contentType"))
                print("    Size      :", attachment.get("size"))
                print("    Inline    :", attachment.get("isInline"))
                print("    ID        :", attachment.get("id"))
                print()
                if (
                    attachment.get("contentType") == "application/pdf"
                    and not attachment.get("isInline")
                ):
                    pdf_bytes = graph_get_bytes(
                        f"/me/messages/{email['id']}"
                        f"/attachments/{attachment['id']}/$value",
                        token,
                    )

                    print("    PDF DOWNLOAD TEST")
                    print("    Bytes received :", len(pdf_bytes))
                    print("    First 8 bytes  :", pdf_bytes[:8])
                    print("    Looks like PDF :", pdf_bytes.startswith(b"%PDF"))
                    print()
                    pdf_text = extract_pdf_text(pdf_bytes)

                    print("    PDF TEXT EXTRACTION")
                    print("    Characters :", len(pdf_text))
                    print("    Preview:")
                    print("    " + "-" * 60)
                    print(pdf_text[:1500])
                    print()
             


if __name__ == "__main__":
    main()