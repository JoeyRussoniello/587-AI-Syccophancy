from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Prompt(Base):
    __tablename__ = "prompts"

    prompt_id = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(Text, nullable=False)
    YTA_NTA = Column(String, nullable=False)
    Flipped = Column(Boolean, nullable=False, default=False)
    Validation = Column(Boolean, nullable=False, default=False)

    responses = relationship("LLMResponse", back_populates="prompt")


class SystemPrompt(Base):
    __tablename__ = "system_prompts"

    system_prompt_id = Column(Integer, primary_key=True, autoincrement=True)
    system_prompt = Column(Text, nullable=False)

    responses = relationship("LLMResponse", back_populates="system_prompt")


class LLMResponse(Base):
    __tablename__ = "llm_responses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey("prompts.prompt_id"), nullable=False)
    system_prompt_id = Column(
        Integer, ForeignKey("system_prompts.system_prompt_id"), nullable=False
    )
    model = Column(String, nullable=False)
    response = Column(Text, nullable=False)

    prompt = relationship("Prompt", back_populates="responses")
    system_prompt = relationship("SystemPrompt", back_populates="responses")
