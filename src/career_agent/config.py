from pydantic import BaseModel


class Settings(BaseModel):
    trusted_senders: set[str] = {
        "nustalentconnect@csm.symplicity.com",
        "no-reply@kinobi.asia",
    }
