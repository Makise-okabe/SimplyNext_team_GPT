# M8 implementation will go here.
# This connector will use delegated Microsoft Graph access.

from career_agent.connectors.base import EmailConnector
from career_agent.models.email import EmailMessage


class OutlookGraphConnector(EmailConnector):
    def get_messages(self) -> list[EmailMessage]:
        raise NotImplementedError("M8: implement Microsoft Graph connector")
