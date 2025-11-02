import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Iterable, List, Optional, Type, TypeVar, cast

from llama_index.core.bridge.pydantic import BaseModel, Field, GenericModel
from llama_index.core.readers.base import BasePydanticReader, ReaderConfig
from llama_index.core.schema import BaseComponent, Document, TextNode


class DataSource(BaseModel):
    """
    A class containing metadata for a type of data source.
    """

    name: str = Field(
        description="Unique and human-readable name for the type of data source"
    )
    component_type: Type[BaseComponent] = Field(
        description="Type of component that implements the data source"
    )


class DocumentGroup(BasePydanticReader):
    """
    A group of documents, usually separate pages from a single file.
    """

    file_path: str = Field(description="Path to the file containing the documents")
    documents: List[Document] = Field(
        description="Sequential group of documents, usually separate pages from a single file."
    )

    @property
    def file_name(self) -> str:
        return Path(self.file_path).name

    @classmethod
    def class_name(cls) -> str:
        return "DocumentGroup"

    def lazy_load_data(self, *args: Any, **load_kwargs: Any) -> Iterable[Document]:
        """Load data from the input directory lazily."""
        return self.documents


def build_conifurable_data_source_enum():
    """
    Build an enum of configurable data sources.
    But conditional on if the corresponding reader is available.
    """


    # Helper function to attempt import and collect enum members efficiently
    def _try_import_and_append(module_path: str, attr_name: str, enum_name: str, display_name: str):
        try:
            # Using __import__ is slightly faster and avoids repeated syntax tree parsing than "from x import y" for many members
            mod = __import__(module_path, fromlist=(attr_name,))
            enum_members.append(
                (
                    enum_name,
                    DataSource(
                        name=display_name,
                        component_type=getattr(mod, attr_name),
                    ),
                )
            )
        except ImportError:
            pass

    class ConfigurableComponent(Enum):
        @classmethod
        def from_component(cls, component: BaseComponent) -> "ConfigurableDataSources":
            component_class = type(component)
            for component_type in cls:
                if component_type.value.component_type == component_class:
                    return component_type
            raise ValueError(
                f"Component {component} is not a supported data source component."
            )

        def build_configured_data_source(
            self, component: BaseComponent, name: Optional[str] = None
        ) -> "ConfiguredDataSource":
            component_type = self.value.component_type
            if not isinstance(component, component_type):
                raise ValueError(
                    f"The enum value {self} is not compatible with component of "
                    f"type {type(component)}"
                )
            elif isinstance(component, BasePydanticReader):
                reader_config = ReaderConfig(loader=component)
                return ConfiguredDataSource[ReaderConfig](
                    component=reader_config
                )  # type: ignore

            if isinstance(component, DocumentGroup) and name is None:
                # if the component is a DocumentGroup, we want to use the
                # full file path as the name of the data source
                component = cast(DocumentGroup, component)
                name = component.file_path

            if name is None:
                suffix = uuid.uuid1()
                name = self.value.name + f" [{suffix}]]"
            return ConfiguredDataSource[component_type](  # type: ignore
                component=component, name=name
            )

    enum_members = []


    # Group all web readers for faster import (reduce repeated imports)
    _try_import_and_append("llama_index.readers.discord", "DiscordReader", "DISCORD", "Discord")
    _try_import_and_append("llama_index.readers.elasticsearch", "ElasticsearchReader", "ELASTICSEARCH", "Elasticsearch")
    _try_import_and_append("llama_index.readers.notion", "NotionPageReader", "NOTION_PAGE", "Notion Page")
    _try_import_and_append("llama_index.readers.slack", "SlackReader", "SLACK", "Slack")
    _try_import_and_append("llama_index.readers.twitter", "TwitterTweetReader", "TWITTER", "Twitter")

    try:
        web_mod = __import__("llama_index.readers.web", fromlist=("SimpleWebPageReader", "TrafilaturaWebReader", "BeautifulSoupWebReader", "RssReader"))
    except ImportError:
        web_mod = None

    if web_mod is not None:
        # Only perform hasattr checks instead of individual try/except for each reader
        for reader_attr, enum_name, display_name in [
            ("SimpleWebPageReader", "SIMPLE_WEB_PAGE", "Simple Web Page"),
            ("TrafilaturaWebReader", "TRAFILATURA_WEB_PAGE", "Trafilatura Web Page"),
            ("BeautifulSoupWebReader", "BEAUTIFUL_SOUP_WEB_PAGE", "Beautiful Soup Web Page"),
            ("RssReader", "RSS", "RSS"),
        ]:
            if hasattr(web_mod, reader_attr):
                enum_members.append(
                    (
                        enum_name,
                        DataSource(
                            name=display_name,
                            component_type=getattr(web_mod, reader_attr),
                        ),
                    )
                )

    _try_import_and_append("llama_index.readers.wikipedia", "WikipediaReader", "WIKIPEDIA", "Wikipedia")
    _try_import_and_append("llama_index.readers.youtube_transcript", "YoutubeTranscriptReader", "YOUTUBE_TRANSCRIPT", "Youtube Transcript")

    # Batch import for Google Readers
    try:
        google_mod = __import__("llama_index.readers.google", fromlist=("GoogleDocsReader", "GoogleSheetsReader"))
    except ImportError:
        google_mod = None

    if google_mod is not None:
        if hasattr(google_mod, "GoogleDocsReader"):
            enum_members.append(
                (
                    "GOOGLE_DOCS",
                    DataSource(
                        name="Google Docs",
                        component_type=getattr(google_mod, "GoogleDocsReader"),
                    ),
                )
            )
        if hasattr(google_mod, "GoogleSheetsReader"):
            enum_members.append(
                (
                    "GOOGLE_SHEETS",
                    DataSource(
                        name="Google Sheets",
                        component_type=getattr(google_mod, "GoogleSheetsReader"),
                    ),
                )
            )

    enum_members.append(
        (
            "READER",
            DataSource(
                name="Reader",
                component_type=ReaderConfig,
            ),
        )
    )

    enum_members.append(
        (
            "DOCUMENT_GROUP",
            DataSource(
                name="Document Group",
                component_type=DocumentGroup,
            ),
        )
    )

    enum_members.append(
        (
            "TEXT_NODE",
            DataSource(
                name="Text Node",
                component_type=TextNode,
            ),
        )
    )

    enum_members.append(
        (
            "DOCUMENT",
            DataSource(
                name="Document",
                component_type=Document,
            ),
        )
    )

    return ConfigurableComponent("ConfigurableDataSources", enum_members)


ConfigurableDataSources = build_conifurable_data_source_enum()

T = TypeVar("T", bound=BaseComponent)


class ConfiguredDataSource(GenericModel, Generic[T]):
    """
    A class containing metadata & implementation for a data source in a pipeline.
    """

    name: str
    component: T = Field(description="Component that implements the data source")

    @classmethod
    def from_component(
        cls, component: BaseComponent, name: Optional[str] = None
    ) -> "ConfiguredDataSource":
        """
        Build a ConfiguredDataSource from a component.

        This should be the preferred way to build a ConfiguredDataSource
        as it will ensure that the component is supported as indicated by having a
        corresponding enum value in DataSources.

        This has the added bonus that you don't need to specify the generic type
        like ConfiguredDataSource[Document]. The return value of
        this ConfiguredDataSource.from_component(document) will be
        ConfiguredDataSource[Document] if document is
        a Document object.
        """
        return ConfigurableDataSources.from_component(
            component
        ).build_configured_data_source(component, name)

    @property
    def configurable_data_source_type(self) -> ConfigurableDataSources:
        return ConfigurableDataSources.from_component(self.component)
