from __future__ import annotations
from dataclasses import fields, Field
from typing import (Any, ClassVar, Dict, ItemsView, List, Protocol, ValuesView, TypeVar,
                    runtime_checkable)

@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Field[Any]]]

D = TypeVar("D", bound="DataContainer")

class DataContainer(DataclassInstance):
    """Abstract data container class based on :class:`dataclass`. Has :class:`dict` interface,
    and :func:`DataContainer.replace` to create a new obj with a set of data attributes replaced.
    """
    def __getitem__(self, attr: str) -> Any:
        return self.__getattribute__(attr)

    def contents(self) -> List[str]:
        """Return a list of the attributes stored in the container that are initialised.

        Returns:
            List of the attributes stored in the container.
        """
        return [attr for attr in self.keys() if self.get(attr) is not None]

    def get(self, attr: str, value: Any=None) -> Any:
        """Retrieve a dataset, return ``value`` if the attribute is not found.

        Args:
            attr : Data attribute.
            value : Data which is returned if the attribute is not found.

        Returns:
            Attribute's data stored in the container, ``value`` if ``attr`` is not found.
        """
        if attr in self.keys():
            return self[attr]
        return value

    def keys(self) -> List[str]:
        """Return a list of the attributes available in the container.

        Returns:
            List of the attributes available in the container.
        """
        return [field.name for field in fields(self)]

    def values(self) -> ValuesView:
        """Return the attributes' data stored in the container.

        Returns:
            List of data stored in the container.
        """
        return dict(self).values()

    def items(self) -> ItemsView:
        """Return (key, value) pairs of the datasets stored in the container.

        Returns:
            (key, value) pairs of the datasets stored in the container.
        """
        return dict(self).items()

    def replace(self: D, **kwargs: Any) -> D:
        """Return a new container object with a set of attributes replaced.

        Args:
            kwargs : A set of attributes and the values to to replace.

        Returns:
            A new container object with updated attributes.
        """
        return type(self)(**dict(self, **kwargs))

    def to_dict(self) -> Dict[str, Any]:
        """Export the :class:`Sample` object to a :class:`dict`.

        Returns:
            A dictionary of :class:`Sample` object's attributes.
        """
        return {attr: self.get(attr) for attr in self.contents()}
