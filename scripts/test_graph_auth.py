from career_agent.connectors.outlook_graph import OutlookGraphConnector


def main() -> None:
    connector = OutlookGraphConnector()
    messages = connector.get_messages(top=20)

    print(f"\nFetched {len(messages)} matching career emails.\n")

    for index, message in enumerate(messages, start=1):
        print(f"[{index}] {message.received_at} | {message.sender_email}")
        print(f"    {message.subject}")
        print(f"    body chars: {len(message.body_text or message.body_html)}")
        print(f"    outlook link: {message.links[0] if message.links else 'none'}")
        print()


if __name__ == "__main__":
    main()
