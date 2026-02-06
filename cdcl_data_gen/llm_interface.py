"""
LLM Interface protocol for the CDCL environment.
"""

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class LLMInterface(Protocol):
    """
    Protocol defining the interface between the environment and an LLM.

    The environment calls generate() to get tokens from the LLM,
    and inject() to provide data (e.g., after READ commands).
    """

    def generate(self, prompt: str) -> Iterator[str]:
        """
        Generate tokens from the LLM.

        Args:
            prompt: The initial prompt or context.

        Yields:
            Tokens one at a time.
        """
        ...

    def inject(self, content: str) -> None:
        """
        Inject content from the environment.

        Called when the environment needs to provide data to the LLM,
        such as after a READ command or at the start of a procedure.

        For mock LLM: validates that the injected content matches the trace.
        For real LLM: appends content to the generation context.

        Args:
            content: The content to inject.

        Raises:
            TraceMismatchError: If content doesn't match expected (mock mode).
        """
        ...

    def start_subcall(self, procedure: str) -> None:
        """
        Signal the start of a subcall.

        Called when the LLM outputs CALL <procedure>.
        For mock LLM: switches to the subcall trace.

        Args:
            procedure: The procedure being called (e.g., "unit_propagate").
        """
        ...

    def end_subcall(self) -> None:
        """
        Signal the end of a subcall.

        Called when *_END pattern is detected.
        For mock LLM: returns to parent trace.
        """
        ...
