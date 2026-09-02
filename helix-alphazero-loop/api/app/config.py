from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)
    cors_origins: str = Field(default="http://localhost:3000", validation_alias=AliasChoices("HELIX_CORS_ORIGINS", "cors_origins"))

    @property
    def cors_origin_list(self) -> list[str]:
        return [p.strip() for p in self.cors_origins.split(",") if p.strip()]


settings = Settings()
