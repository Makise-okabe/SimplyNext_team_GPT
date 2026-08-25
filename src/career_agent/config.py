from pydantic import BaseModel


class Settings(BaseModel):
    trusted_senders: set[str] = {
        "zeli.goh@nus.edu.sg",
        "no-reply@kinobi.asia",
    }
