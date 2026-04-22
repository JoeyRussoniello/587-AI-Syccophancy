from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base declarative class for all ORM models in the project."""

    pass


class Prompt(Base):
    """Stored prompt text and its ground-truth AITA label metadata."""

    __tablename__ = "prompts"

    prompt_id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text, nullable=False)
    top_comment = Column(Text, nullable=True)
    YTA_NTA = Column(String, nullable=False)
    Flipped = Column(Boolean, nullable=False, default=False)
    Validation = Column(Boolean, nullable=False, default=False)

    responses = relationship("LLMResponse", back_populates="prompt")


class SystemPrompt(Base):
    """Normalized system prompt values referenced by model responses."""

    __tablename__ = "system_prompts"

    system_prompt_id = Column(Integer, primary_key=True, autoincrement=True)
    system_prompt_name = Column(String, nullable=False)
    system_prompt = Column(Text, nullable=False)

    responses = relationship("LLMResponse", back_populates="system_prompt")


class LLMResponse(Base):
    """A model response tied to a prompt, system prompt, and extracted label."""

    __tablename__ = "llm_responses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey("prompts.prompt_id"), nullable=False)
    system_prompt_id = Column(
        Integer, ForeignKey("system_prompts.system_prompt_id"), nullable=False
    )
    model = Column(String, nullable=False)
    llm_label = Column(String, nullable=True)
    response = Column(Text, nullable=False)

    prompt = relationship("Prompt", back_populates="responses")
    system_prompt = relationship("SystemPrompt", back_populates="responses")
