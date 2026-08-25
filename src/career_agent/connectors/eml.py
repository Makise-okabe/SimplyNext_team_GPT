# M1 implementation will go here.
# It will parse local .eml files into EmailMessage objects.

from career_agent.connectors.base import EmailConnector
from career_agent.models.email import EmailMessage


class EmlConnector(EmailConnector):
    def get_messages(self) -> list[EmailMessage]:
        raise NotImplementedError("M1: implement EML parsing")
