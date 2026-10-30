
from __future__ import annotations

from dataclasses import dataclass


QUOTE_TOOL = "get_mcx_quote"
CALCULATION_TOOL = "calculate_mcx_contract_value"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    keywords: tuple[str, ...]
    instructions: str
    allowed_tools: frozenset[str]


@dataclass(frozen=True)
class SkillSelection:
    skills: tuple[Skill, ...]
    allowed_tools: frozenset[str]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(skill.name for skill in self.skills)

    @property
    def instructions(self) -> str:
        if not self.skills:
            return (
                "General finance skill: answer educational and conceptual "
                "questions clearly. Distinguish facts from assumptions. "
                "Do not invent current market data."
            )

        sections = []

        for skill in self.skills:
            sections.append(
                f"Skill: {skill.name}\n"
                f"Purpose: {skill.description}\n"
                f"Instructions: {skill.instructions}"
            )

        return "\n\n".join(sections)


class SkillRegistry:
    """Select domain skills using deterministic keyword matching."""

    def __init__(self) -> None:
        self._skills = (
            Skill(
                name="commodity_research",
                description="Research commodity prices and market information.",
                keywords=(
                    "commodity", "mcx", "gold", "silver", "crude oil",
                    "natural gas", "copper", "bhavcopy", "quote",
                ),
                instructions=(
                    "Use imported market data only when relevant. "
                    "Identify the source and observation date. "
                    "Never describe end-of-day data as live data."
                ),
                allowed_tools=frozenset({QUOTE_TOOL}),
            ),
            Skill(
                name="market_analysis",
                description="Explain commodity market movements and benchmarks.",
                keywords=(
                    "market analysis", "market trend", "price movement",
                    "benchmark", "spread", "price curve", "market outlook",
                    "fundamental analysis", "technical analysis",
                ),
                instructions=(
                    "Separate historical observations, market drivers, "
                    "and forward-looking assumptions. Do not invent "
                    "market indicators or prices."
                ),
                allowed_tools=frozenset({QUOTE_TOOL}),
            ),
            Skill(
                name="risk_analysis",
                description="Explain commodity risk, hedging, and valuation.",
                keywords=(
                    "risk", "hedge", "hedging", "exposure", "var",
                    "volatility", "notional", "contract value",
                    "calculate", "margin", "risk management",
                ),
                instructions=(
                    "Explain assumptions and relevant units. "
                    "Use calculation tools only when their validation "
                    "requirements are satisfied. Distinguish notional "
                    "value from margin, profit, and loss."
                ),
                allowed_tools=frozenset({
                    QUOTE_TOOL,
                    CALCULATION_TOOL,
                }),
            ),
            Skill(
                name="portfolio_analysis",
                description="Explain investment portfolios and allocation.",
                keywords=(
                    "portfolio", "allocation", "diversification",
                    "investment", "investor", "portfolio return",
                    "portfolio performance",
                ),
                instructions=(
                    "Discuss diversification, concentration, risk, "
                    "and assumptions. Do not fabricate portfolio "
                    "holdings or guarantee investment returns."
                ),
                allowed_tools=frozenset(),
            ),
            Skill(
                name="trading_workflow",
                description="Explain commodity trading and contract workflows.",
                keywords=(
                    "trade lifecycle", "trading workflow", "settlement",
                    "physical delivery", "futures contract",
                    "forward contract", "position management",
                    "trade confirmation",
                ),
                instructions=(
                    "Explain workflows and operational controls. "
                    "You may describe trade execution concepts, but "
                    "you cannot place, amend, or cancel trades."
                ),
                allowed_tools=frozenset({QUOTE_TOOL}),
            ),
        )

    def list_skills(self) -> tuple[Skill, ...]:
        return self._skills

    def get_skill(self, name: str) -> Skill | None:
        return next(
            (skill for skill in self._skills if skill.name == name),
            None,
        )

    def select(self, user_message: str) -> SkillSelection:
        """Select matching skills and combine their tool permissions."""
        normalized = user_message.casefold()

        selected = [
            skill
            for skill in self._skills
            if any(
                keyword in normalized
                for keyword in skill.keywords
            )
        ]

        allowed_tools = frozenset().union(
            *(skill.allowed_tools for skill in selected)
        ) if selected else frozenset()

        return SkillSelection(
            skills=tuple(selected),
            allowed_tools=allowed_tools,
        )
