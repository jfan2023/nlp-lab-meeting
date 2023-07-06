class CustomSpan:
    """
    Custom Span class, which allows for easier equality/range checks.
    """

    start: int
    end: int

    def __init__(self, start: int, end: int):
        if end < start:
            raise ValueError(
                f"Span cannot be initialized for negative range! `end` must be larger or equal to `start`"
            )
        self.start = start
        self.end = end

    def __contains__(self, item):
        """
        Overload behavior for checks like
            CustomSpan(0, 2) in CustomSpan(-1, 3)
        """
        if isinstance(item, CustomSpan):
            if item.start >= self.start and item.end <= self.end:
                return True
            else:
                return False
        elif isinstance(item, tuple) or isinstance(item, list):
            if item[0] >= self.start and item[1] <= self.end:
                return True
            else:
                return False
        else:
            raise NotImplementedError(
                f"Comparison between CustomSpan and {type(item)} not supported!"
            )

    def __len__(self):
        """
        Define the "length" of a span.
        """
        return self.end - self.start

    def __repr__(self):
        """
        Surface form representation.
        """
        return f"({self.start}, {self.end})"

    def __eq__(self, other):
        """
        Overload comparison functionality, allowing for checks with other CustomSpans and tuples/lists
        """
        if isinstance(other, CustomSpan):
            if other.start == self.start and other.end == self.end:
                return True
            else:
                return False
        elif isinstance(other, tuple) or isinstance(other, list):
            if other[0] == self.start and other[1] == self.end:
                return True
            else:
                return False
        else:
            return NotImplementedError(
                f"Comparison between CustomSpan and {type(other)} not supported!"
            )

    def __hash__(self):
        """
        Once __eq__ is defined, __hash__ also needs to be re-defined to avoid `Unhashable` errors.
        """
        return hash((self.start, self.end))