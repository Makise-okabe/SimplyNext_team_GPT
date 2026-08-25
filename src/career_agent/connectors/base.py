from abc import ABC, abstractmethod
from career_agent.models.email import EmailMessage


class EmailConnector(ABC):
    @abstractmethod
    def get_messages(self) -> list[EmailMessage]:
        raise NotImplementedError
