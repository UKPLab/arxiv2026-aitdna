from enum import Enum


class Notion(Enum):
    # base
    SPAN_LEVEL = "SPAN_LEVEL"
    TOKEN_LEVEL = "TOKEN_LEVEL"
    # genetic
    DOCUMENT_LEVEL = "DOCUMENT_LEVEL"
    SENTENCE_LEVEL = "SENTENCE_LEVEL"
    BOUNDARY_LEVEL = "BOUNDARY_LEVEL"
    CONTENT_BASED = "CONTENT_BASED"
    INTENT_BASED = "INTENT_BASED"

    # population
    MEMBERSHIP_BASED = "MEMBERSHIP_BASED"
    AUTHORSHIP_BASED = "AUTHORSHIP_BASED"

    @staticmethod
    def get(val: str):
        """Return a tuple (A member, Category) for the given value, or None if not found."""
        value_to_member = {member.value: member for member in Notion.__members__.values()}

        member = value_to_member[val]
        if member is None:
            return None

        return member

    def is_genetic(self):
        return self in [Notion.MEMBERSHIP_BASED, Notion.AUTHORSHIP_BASED]