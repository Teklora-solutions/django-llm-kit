"""Touchpoints: each place your code calls an LLM, declared once with its model policy.

    from llm_kit import touchpoint, ClientChoice

    summarise = touchpoint(
        "crm.summarise_lead",
        default="anthropic/claude-haiku-5-5",
        fallbacks=["gemini/gemini-3.8-flash"],
        params={"temperature": 0},
        requires={"structured_output": True},
    )

    llm = summarise.model()                       # LangChain chat model (with fallbacks)
    lead = summarise.complete("...", schema=Lead) # one-shot structured call

In a graph node, pass the node's ``config`` so the run's requested model
(``configurable.model``) and tracing metadata flow through::

    def node(state, config):
        return {"messages": [chat.model(config=config).invoke(state["messages"], config)]}
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import BaseModel

from llm_kit.core.catalogue.match import REQUIREMENT_KEYS
from llm_kit.core.errors import ConfigError, UnknownTouchpoint
from llm_kit.core.touchpoints.env import ResolutionEnv, default_env
from llm_kit.core.touchpoints.policy import Choice, ServerChoice
from llm_kit.core.touchpoints.resolve import Resolution, resolve
from llm_kit.core.touchpoints.retry import RetryPolicy, policy_for
from llm_kit.core.touchpoints.structured import parse_as

_registry: dict[str, Touchpoint] = {}
_lock = threading.Lock()


def _requested_from(config: RunnableConfig | None) -> str | None:
    return ((config or {}).get("configurable") or {}).get("model")


def _messages(prompt: str | Sequence[BaseMessage] | Sequence[Any]) -> list:
    return [HumanMessage(content=prompt)] if isinstance(prompt, str) else list(prompt)


@dataclass(frozen=True, eq=False)
class Touchpoint:
    name: str
    default: str
    fallbacks: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    choice: Choice = field(default_factory=ServerChoice)
    requires: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    backend: str | None = None  # None = LLM_KIT["MODEL_BACKEND"]
    capture: bool | None = None  # store prompts/responses; None = LLM_KIT["CAPTURE_CONTENT"]
    retry: RetryPolicy | None = None  # waits before falling back; None = LLM_KIT["RETRY"]
    daily_budget: Any = None  # USD per day across all its calls; None = no limit
    settable: Any = None  # a core.scoped.Settable: what tenants and users may save for it
    embedding_backend: Any = None  # embed()'s backend when it differs from backend=; None = backend=
    # fn(input, ref, info) -> input, called for each model just before it is tried (also after a
    # fallback), so each gets input it can read; a dotted path or a callable.
    adapt_input: Any = None
    # {ref or pattern: params} applied to that model only, over ``params`` (voices, sizes and
    # aspect ratios differ by provider); patterns like "openai/*", the most specific last.
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    media_backend: Any = None  # media calls' backend; None = LLM_KIT["MEDIA_BACKEND"]
    fallback_on_refusal: bool = False  # a media call refused under a content policy tries the next model

    def __post_init__(self) -> None:
        bad = [k for k in self.requires if k not in REQUIREMENT_KEYS]
        if bad:
            raise ConfigError(
                f"Touchpoint {self.name!r}: unknown requirement(s) {bad}. Use: {', '.join(REQUIREMENT_KEYS)}."
            )
        if isinstance(self.fallbacks, str):
            object.__setattr__(self, "fallbacks", (self.fallbacks,))
        object.__setattr__(self, "retry", RetryPolicy.coerce(self.retry))

    def params_for(self, ref: str, params: dict[str, Any]) -> dict[str, Any]:
        """``params`` with this touchpoint's ``model_params`` for ``ref`` on top."""
        if not self.model_params:
            return dict(params)
        import fnmatch

        from llm_kit.core.touchpoints.embeddings import _specificity

        out = dict(params)
        matching = [p for p in self.model_params if p == ref or fnmatch.fnmatchcase(ref, p)]
        for pattern in sorted(matching, key=_specificity):
            out.update(self.model_params[pattern] or {})
        return out

    # ----------------------------------------------------------- resolution

    def resolve(
        self, requested: str | None = None, *, config: RunnableConfig | None = None, env: ResolutionEnv | None = None
    ) -> Resolution:
        """Which model(s) a call would use, and why. Raises ``ModelNotAllowed`` / ``TouchpointDisabled``."""
        return resolve(self, requested if requested is not None else _requested_from(config), self.env_for(config, env))

    def env_for(self, config: RunnableConfig | None = None, env: ResolutionEnv | None = None) -> ResolutionEnv:
        """The env a call uses: for a known caller, one that applies what their tenant and user
        saved and the keys they brought (an env with ``for_caller``, as the Django adapter's has)."""
        env = env or default_env()
        for_caller = getattr(env, "for_caller", None)
        if for_caller is None:
            return env
        run = config if config is not None else _running()
        caller = ((run or {}).get("configurable") or {}).get("__llm_kit_caller")
        if caller is None:
            return env
        scoped = for_caller(caller)
        for_touchpoint = getattr(scoped, "for_touchpoint", None)  # key checks follow this touchpoint's policy
        return for_touchpoint(self) if for_touchpoint is not None else scoped

    # --------------------------------------------------------------- models

    def model(
        self,
        requested: str | None = None,
        *,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        adapt_input: Any = None,
        **params: Any,
    ) -> Runnable:
        """A chat model for this touchpoint, wrapped with its fallbacks.

        ``params`` override the touchpoint's params for this call (e.g. ``max_tokens``).
        ``retry`` overrides how long a short rate limit is waited out before falling back.
        ``fallbacks=False`` uses only the resolved model for this call (results compared
        across runs shouldn't come from whichever model answered); it still waits per ``retry``.
        The returned runnable supports ``invoke``/``stream``/``bind_tools``/``with_structured_output``.
        ``caller`` (a ``Caller``, an ``Owner`` or a user) is who the call is for, outside a request or
        agent run: their saved settings, keys and budgets apply (invoke with ``run_config(config)``
        so it is recorded for them too). ``adapt_input`` replaces the touchpoint's for this call
        (``False``: none).
        """
        from llm_kit.core.observability.spend import check_daily_budget
        from llm_kit.core.touchpoints.backends import build_for

        config = with_caller(config, caller)
        check_daily_budget(self)  # BudgetExceeded once today's spend reaches daily_budget
        env = self.env_for(config, env)
        if (check_budgets := getattr(env, "check_budgets", None)) is not None:
            check_budgets(self)  # the caller's tenant and user budgets
        resolution = self.resolve(requested, config=config, env=env)
        if not fallbacks:
            resolution.fallbacks = []
        if params:
            resolution.params = {**resolution.params, **params}
        # Inside a running agent, its retry policy applies (but not its requested model).
        run = config if config is not None else _running()
        return build_for(
            self, resolution, env or default_env(), retry=policy_for(self, run, retry), adapt_input=adapt_input
        )

    # ------------------------------------------------------- one-shot calls

    def complete(
        self,
        prompt: str | Sequence[BaseMessage] | Sequence[Any],
        *,
        schema: type[BaseModel] | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        **params: Any,
    ) -> Any:
        """Call the model once. With ``schema``, return a validated instance of it.

        Uses native structured output when the model supports it, otherwise
        parses JSON out of the reply (code fences and surrounding prose are tolerated).
        """
        config = with_caller(config, caller)
        llm = self.model(requested, config=config, env=env, retry=retry, fallbacks=fallbacks, **params)
        messages = _messages(prompt)
        run_config = self._run_config(config)
        if schema is None:
            return _as_run(llm).invoke(messages, run_config)
        structured = _structured(llm, schema)
        if structured is not None:
            return _parsed(structured.invoke(messages, run_config), schema)
        reply = _as_run(llm).invoke(messages, run_config)
        return parse_as(_text(reply), schema)

    async def acomplete(
        self,
        prompt: str | Sequence[BaseMessage] | Sequence[Any],
        *,
        schema: type[BaseModel] | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        **params: Any,
    ) -> Any:
        config = with_caller(config, caller)
        llm = self.model(requested, config=config, env=env, retry=retry, fallbacks=fallbacks, **params)
        messages = _messages(prompt)
        run_config = self._run_config(config)
        if schema is None:
            return await _as_run(llm).ainvoke(messages, run_config)
        structured = _structured(llm, schema)
        if structured is not None:
            return _parsed(await structured.ainvoke(messages, run_config), schema)
        reply = await _as_run(llm).ainvoke(messages, run_config)
        return parse_as(_text(reply), schema)

    # ----------------------------------------------------------- embeddings

    def embed(
        self,
        inputs: Any,
        *,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        **params: Any,
    ) -> list[list[float]]:
        """Embed ``inputs`` (a string, or a list of strings or media parts): one vector each.

        Same resolution, keys, fallbacks, retry waits and call records as chat calls; see
        ``llm_kit.core.touchpoints.embeddings``. ``params`` (e.g. ``dimensions``) go to the backend."""
        from llm_kit.core.touchpoints import embeddings

        call, items = embeddings.prepare(
            self,
            inputs,
            requested=requested,
            config_=with_caller(config, caller),
            env=env,
            retry=retry,
            fallbacks=fallbacks,
            params={},
        )
        return embeddings.run_sync(call, items, params)

    async def aembed(
        self,
        inputs: Any,
        *,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        **params: Any,
    ) -> list[list[float]]:
        from llm_kit.core.touchpoints import embeddings

        call, items = embeddings.prepare(
            self,
            inputs,
            requested=requested,
            config_=with_caller(config, caller),
            env=env,
            retry=retry,
            fallbacks=fallbacks,
            params={},
        )
        return await embeddings.run_async(call, items, params)

    # ---------------------------------------------------------------- media

    def _media(self, kind, requested=None, config=None, env=None, retry=None, fallbacks=True, caller=None):
        from llm_kit.core.media import calls
        from llm_kit.core.media.kinds import get_kind

        return calls.prepare(
            self,
            get_kind(kind),
            requested=requested,
            config_=with_caller(config, caller),
            env=env,
            retry=retry,
            fallbacks=fallbacks,
        )

    def generate(
        self,
        kind: str,
        *,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        wait: float | None = None,
        params: dict[str, Any] | None = None,
        **request: Any,
    ) -> Any:
        """Make (or read) media of ``kind`` (``"image"``, ``"video"``, ``"speech"``, a kind you
        registered...) from ``request`` (``prompt=``, ``text=``, ``audio=``...). Returns a
        ``MediaResult``. ``params`` go to the provider as they are. The kind's own method
        (``image()``, ``speak()``...) is easier where there is one. See ``llm_kit.core.media``."""
        from llm_kit.core.media import calls

        call = self._media(kind, requested, config, env, retry, fallbacks, caller)
        return calls.run_sync(call, request, dict(params or {}), wait)

    async def agenerate(
        self,
        kind: str,
        *,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        wait: float | None = None,
        params: dict[str, Any] | None = None,
        **request: Any,
    ) -> Any:
        from llm_kit.core.media import calls

        call = self._media(kind, requested, config, env, retry, fallbacks, caller)
        return await calls.run_async(call, request, dict(params or {}), wait)

    def _run_media(self, kind: str, request: dict[str, Any], options: dict[str, Any], params: dict[str, Any]):
        from llm_kit.core.media import calls

        wait, fetch = options.pop("wait", None), options.pop("fetch", True)
        call = self._media(kind, **options)
        result = calls.run_sync(call, _drop_none(request), params, wait)
        return calls.fetched(result) if fetch else result

    async def _arun_media(self, kind: str, request: dict[str, Any], options: dict[str, Any], params: dict[str, Any]):
        from llm_kit.core.media import calls

        wait, fetch = options.pop("wait", None), options.pop("fetch", True)
        call = self._media(kind, **options)
        result = await calls.run_async(call, _drop_none(request), params, wait)
        return await calls.afetched(result) if fetch else result

    def image(
        self,
        prompt: str,
        *,
        n: int | None = None,
        size: str | None = None,
        quality: str | None = None,
        style: str | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        fetch: bool = True,
        **params: Any,
    ) -> Any:
        """Make ``n`` images from ``prompt``; a ``MediaResult`` whose ``media`` are the images.

        ``size`` (``"1024x1024"``), ``quality`` and ``style`` go to models that take them; other
        ``params`` (``aspect_ratio``, ``output_format``, ``background``...) go to the provider as
        they are. Images the provider returns as URLs are downloaded (``fetch=False``: keep the
        URLs; they often expire within the hour). See the media guide."""
        request = {"prompt": prompt, "n": n, "size": size, "quality": quality, "style": style}
        options = _options(requested, config, env, retry, fallbacks, caller, fetch=fetch)
        return self._run_media("image", request, options, params)

    async def aimage(
        self,
        prompt: str,
        *,
        n: int | None = None,
        size: str | None = None,
        quality: str | None = None,
        style: str | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        fetch: bool = True,
        **params: Any,
    ) -> Any:
        request = {"prompt": prompt, "n": n, "size": size, "quality": quality, "style": style}
        options = _options(requested, config, env, retry, fallbacks, caller, fetch=fetch)
        return await self._arun_media("image", request, options, params)

    def edit_image(
        self,
        prompt: str,
        images: Any,
        *,
        mask: Any = None,
        n: int | None = None,
        size: str | None = None,
        quality: str | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        fetch: bool = True,
        **params: Any,
    ) -> Any:
        """Change ``images`` (one, or a list: bytes, paths, URLs, files or ``Media``) as ``prompt``
        says; ``mask`` marks where (transparent pixels are changed). A ``MediaResult``."""
        request = _edit_request(prompt, images, mask, n, size, quality)
        options = _options(requested, config, env, retry, fallbacks, caller, fetch=fetch)
        return self._run_media("image_edit", request, options, params)

    async def aedit_image(
        self,
        prompt: str,
        images: Any,
        *,
        mask: Any = None,
        n: int | None = None,
        size: str | None = None,
        quality: str | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        fetch: bool = True,
        **params: Any,
    ) -> Any:
        request = _edit_request(prompt, images, mask, n, size, quality)
        options = _options(requested, config, env, retry, fallbacks, caller, fetch=fetch)
        return await self._arun_media("image_edit", request, options, params)

    def speak(
        self,
        text: str,
        *,
        voice: Any = None,
        format: str | None = None,
        speed: float | None = None,
        instructions: str | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        **params: Any,
    ) -> Any:
        """Read ``text`` aloud: a ``MediaResult`` whose ``first`` is the audio.

        ``voice`` is the provider's voice name or id (set it per model with ``model_params``: no
        two providers share voices); ``format`` is ``mp3`` (usual default), ``wav``, ``opus``,
        ``aac``, ``flac`` or ``pcm``; ``instructions`` steer the delivery on models that take them."""
        request = {"text": text, "voice": voice, "format": format, "speed": speed, "instructions": instructions}
        options = _options(requested, config, env, retry, fallbacks, caller)
        return self._run_media("speech", request, options, params)

    async def aspeak(
        self,
        text: str,
        *,
        voice: Any = None,
        format: str | None = None,
        speed: float | None = None,
        instructions: str | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        **params: Any,
    ) -> Any:
        request = {"text": text, "voice": voice, "format": format, "speed": speed, "instructions": instructions}
        options = _options(requested, config, env, retry, fallbacks, caller)
        return await self._arun_media("speech", request, options, params)

    def transcribe(
        self,
        audio: Any,
        *,
        language: str | None = None,
        prompt: str | None = None,
        format: str | None = None,
        timestamps: Any = None,
        temperature: float | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        **params: Any,
    ) -> Any:
        """Write down what is said in ``audio`` (bytes, a URL, a file or ``Media``): a
        ``MediaResult`` whose ``text`` is the transcript, with ``language``, ``duration``,
        ``segments`` and ``words`` when the provider gives them.

        ``language`` (``"en"``) and ``prompt`` (names, spellings) help the model; ``format`` is
        ``json``, ``text``, ``srt``, ``vtt`` or ``verbose_json``; ``timestamps=True`` (or
        ``["word", "segment"]``) asks for timings."""
        request = _transcription_request(audio, language, prompt, format, timestamps, temperature)
        options = _options(requested, config, env, retry, fallbacks, caller)
        return self._run_media("transcription", request, options, params)

    async def atranscribe(
        self,
        audio: Any,
        *,
        language: str | None = None,
        prompt: str | None = None,
        format: str | None = None,
        timestamps: Any = None,
        temperature: float | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        **params: Any,
    ) -> Any:
        request = _transcription_request(audio, language, prompt, format, timestamps, temperature)
        options = _options(requested, config, env, retry, fallbacks, caller)
        return await self._arun_media("transcription", request, options, params)

    def ocr(
        self,
        document: Any,
        *,
        pages: Any = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        **params: Any,
    ) -> Any:
        """Read the text of ``document`` (a PDF or an image: bytes, a URL, a file or ``Media``): a
        ``MediaResult`` whose ``text`` is the whole document in Markdown and ``pages`` each page's
        (``{"index", "markdown"}``). ``pages`` picks pages (``[0, 2]``) on models that take it."""
        request = {"document": _one_media(document), "pages": pages}
        options = _options(requested, config, env, retry, fallbacks, caller)
        return self._run_media("ocr", request, options, params)

    async def aocr(
        self,
        document: Any,
        *,
        pages: Any = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        **params: Any,
    ) -> Any:
        request = {"document": _one_media(document), "pages": pages}
        options = _options(requested, config, env, retry, fallbacks, caller)
        return await self._arun_media("ocr", request, options, params)

    def video(
        self,
        prompt: str,
        *,
        image: Any = None,
        seconds: float | None = None,
        size: str | None = None,
        resolution: str | None = None,
        wait: float | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        **params: Any,
    ) -> Any:
        """Make a video from ``prompt`` (and ``image``, its first frame or reference), waiting for
        it: at most ``wait`` seconds (``MEDIA_JOB_WAIT``). Videos take minutes: in a request,
        prefer ``start_video()``, which runs it in the background. A ``MediaResult``."""
        request = _video_request(prompt, image, seconds, size, resolution)
        options = _options(requested, config, env, retry, fallbacks, caller, wait=wait)
        return self._run_media("video", request, options, params)

    async def avideo(
        self,
        prompt: str,
        *,
        image: Any = None,
        seconds: float | None = None,
        size: str | None = None,
        resolution: str | None = None,
        wait: float | None = None,
        requested: str | None = None,
        config: RunnableConfig | None = None,
        env: ResolutionEnv | None = None,
        retry: RetryPolicy | dict | str | None = None,
        fallbacks: bool = True,
        caller: Any = None,
        **params: Any,
    ) -> Any:
        request = _video_request(prompt, image, seconds, size, resolution)
        options = _options(requested, config, env, retry, fallbacks, caller, wait=wait)
        return await self._arun_media("video", request, options, params)

    def start(
        self,
        kind: str,
        *,
        owner: Any = None,
        user: Any = None,
        tenant: Any = None,
        requested: str | None = None,
        wait: float | None = None,
        params: dict[str, Any] | None = None,
        **request: Any,
    ) -> Any:
        """Make media of ``kind`` in the background (Django): returns its ``BackgroundRun`` at once,
        to watch, cancel, or hear about through ``media_job_finished``. See ``llm_kit.media.jobs``."""
        from llm_kit.media import jobs

        return jobs.start(
            self,
            kind,
            request,
            params=params,
            requested=requested,
            owner=owner,
            user=user,
            tenant=tenant,
            wait=wait,
        )

    def start_video(
        self,
        prompt: str,
        *,
        image: Any = None,
        seconds: float | None = None,
        size: str | None = None,
        resolution: str | None = None,
        owner: Any = None,
        user: Any = None,
        tenant: Any = None,
        requested: str | None = None,
        wait: float | None = None,
        **params: Any,
    ) -> Any:
        """``video()`` in the background (Django): returns its ``BackgroundRun`` at once."""
        request = _drop_none(_video_request(prompt, image, seconds, size, resolution))
        return self.start(
            "video", owner=owner, user=user, tenant=tenant, requested=requested, wait=wait, params=params, **request
        )

    async def astart(self, kind: str, **kw: Any) -> Any:
        from asgiref.sync import sync_to_async

        return await sync_to_async(self.start)(kind, **kw)

    async def astart_video(self, prompt: str, **kw: Any) -> Any:
        from asgiref.sync import sync_to_async

        return await sync_to_async(self.start_video)(prompt, **kw)

    def run_config(self, config: RunnableConfig | None = None) -> RunnableConfig | None:
        """The config to invoke this touchpoint's ``model()`` with, so the call is recorded.

        Inside a graph node (config already has callbacks) the call joins that run. Called
        without one from inside a running agent (a tool, a code agent, code they call), it
        joins that agent's run too: recorded in its run tree and counted in its totals.
        Otherwise it is a recorded run of its own, named after the touchpoint. Metadata you
        pass (for example ``{"llm_kit_agent": "crm.writer"}``) is recorded with the call, and so
        is the user and tenant of a caller in ``configurable`` (see ``caller=``)."""
        config = _caller_metadata(config)
        if config and config.get("callbacks"):
            return config
        if (parent := _running()) is not None and parent.get("callbacks") is not None:
            mine = config or {}
            return {
                **mine,
                "callbacks": parent["callbacks"],
                "metadata": {**(parent.get("metadata") or {}), **(mine.get("metadata") or {})},
                "configurable": {**(parent.get("configurable") or {}), **(mine.get("configurable") or {})},
                "run_name": mine.get("run_name") or self.name,
            }
        from llm_kit.core.observability.handler import get_handler

        handler = get_handler()
        if handler is None:
            return config
        return {**(config or {}), "callbacks": [handler], "run_name": self.name}

    _run_config = run_config


def _options(requested, config, env, retry, fallbacks, caller, **more) -> dict[str, Any]:
    return {
        "requested": requested,
        "config": config,
        "env": env,
        "retry": retry,
        "fallbacks": fallbacks,
        "caller": caller,
        **more,
    }


def _drop_none(request: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in request.items() if v is not None}


def _media_list(value: Any) -> list:
    from llm_kit.core.media.types import Media

    if isinstance(value, (list, tuple)):
        return [Media.of(v) for v in value]
    return [Media.of(value)]


def _one_media(value: Any) -> Any:
    from llm_kit.core.media.types import Media

    return Media.of(value)


def _edit_request(prompt, images, mask, n, size, quality) -> dict[str, Any]:
    from llm_kit.core.media.types import Media

    return {
        "prompt": prompt,
        "images": _media_list(images),
        "mask": Media.of(mask) if mask is not None else None,
        "n": n,
        "size": size,
        "quality": quality,
    }


def _transcription_request(audio, language, prompt, format, timestamps, temperature) -> dict[str, Any]:
    from llm_kit.core.media.types import Media, audio_seconds

    media = Media.of(audio)
    if timestamps is True:
        timestamps = ["segment"]
    elif isinstance(timestamps, str):
        timestamps = [timestamps]
    return {
        "audio": media,
        "language": language,
        "prompt": prompt,
        "format": format,
        "timestamps": timestamps or None,
        "temperature": temperature,
        "duration": audio_seconds(media) if media.data is not None else None,
    }


def _video_request(prompt, image, seconds, size, resolution) -> dict[str, Any]:
    from llm_kit.core.media.types import Media

    return {
        "prompt": prompt,
        "image": Media.of(image) if image is not None else None,
        "seconds": seconds,
        "size": size,
        "resolution": resolution,
    }


CALLER_KEY = "__llm_kit_caller"  # same as llm_kit.tools.caller.CONFIG_KEY


def with_caller(config: RunnableConfig | None, caller: Any) -> RunnableConfig | None:
    """``config`` with ``caller`` as the run's caller (``caller=None`` leaves it as it is)."""
    if caller is None:
        return config
    config = dict(config or {})
    config["configurable"] = {**(config.get("configurable") or {}), CALLER_KEY: caller}
    return config


def caller_ids(caller: Any) -> tuple[str, str]:
    """``(user id, tenant id)`` as recorded on calls, "" when unknown, for a ``Caller``, an
    ``Owner`` or a user."""
    user = getattr(caller, "user", caller)
    user_id = getattr(user, "pk", None) if getattr(user, "is_authenticated", False) else None
    tenant = getattr(caller, "tenant_id", None)
    return ("" if user_id is None else str(user_id)), ("" if tenant in (None, "") else str(tenant))


def _caller_metadata(config: RunnableConfig | None) -> RunnableConfig | None:
    """Record the configured caller's user and tenant, unless the metadata already says."""
    caller = ((config or {}).get("configurable") or {}).get(CALLER_KEY)
    if caller is None:
        return config
    user_id, tenant = caller_ids(caller)
    metadata = dict(config.get("metadata") or {})
    if user_id and "user_id" not in metadata:
        metadata["user_id"] = user_id
    if tenant and "tenant" not in metadata:
        metadata["tenant"] = tenant
    return {**config, "metadata": metadata}


def _running() -> RunnableConfig | None:
    from llm_kit.core.agents.limits import running_config

    return running_config()


def _as_run(llm: Runnable) -> Runnable:
    """A one-step chain, so the call gets a root run (and a Run record)."""
    from langchain_core.runnables import RunnablePassthrough

    return RunnablePassthrough() | llm


def _structured(llm: Runnable, schema: type[BaseModel]) -> Runnable | None:
    try:
        return llm.with_structured_output(schema, include_raw=True)
    except (NotImplementedError, AttributeError):
        return None


def _parsed(result: dict[str, Any], schema: type[BaseModel]) -> Any:
    """Native structured output can return nothing when the model answers in text; parse the text then."""
    if result.get("parsed") is not None:
        return result["parsed"]
    return parse_as(_text(result["raw"]), schema)


def _text(reply: Any) -> str:
    from llm_kit.core.serialize import message_text

    return message_text(reply) if not isinstance(reply, str) else reply


def touchpoint(
    name: str,
    *,
    default: str,
    fallbacks: Sequence[str] = (),
    params: dict[str, Any] | None = None,
    choice: Choice | None = None,
    requires: dict[str, Any] | None = None,
    description: str = "",
    backend: str | None = None,
    capture: bool | None = None,
    retry: RetryPolicy | dict | str | None = None,
    daily_budget: Any = None,
    settable: Any = None,
    embedding_backend: Any = None,
    adapt_input: Any = None,
    model_params: dict[str, dict[str, Any]] | None = None,
    media_backend: Any = None,
    fallback_on_refusal: bool = False,
) -> Touchpoint:
    """Declare and register a touchpoint. Re-declaring the same name replaces it (module reloads).

    ``daily_budget`` (USD) caps what all its calls may spend in a day; past it, ``model()``,
    ``complete()`` and ``embed()`` raise ``BudgetExceeded`` (``DAILY_BUDGETS["touchpoints"]``
    sets or overrides it without a deploy). ``settable`` (a ``Settable``) lets tenants and users
    save a model choice and parameters for it (see ``llm_kit.scoped``). ``embed()`` uses
    ``embedding_backend``, else ``backend`` (a name, dotted path or factory), else
    ``LLM_KIT["EMBEDDING_BACKEND"]``.

    ``adapt_input`` (``fn(input, ref, info) -> input``, or its dotted path) is called for each
    model just before that model is tried, fallbacks included, with the model's ref and catalogue
    entry (``ModelInfo`` or None): give each model input it can read (audio it can hear, or a
    transcript; a file reference only the provider that holds the file can open). It may be async
    in async calls. If it raises, that model is skipped.

    ``model_params`` (``{ref or pattern: params}``) adds parameters for one model only, over
    ``params``: a voice, size or aspect ratio another provider doesn't have. ``media_backend``
    makes this touchpoint's media (``image()``, ``video()``, ``speak()``...; default
    ``LLM_KIT["MEDIA_BACKEND"]``). ``fallback_on_refusal=True`` lets a media call that a provider
    refused under its content policy try the next model (off by default)."""
    tp = Touchpoint(
        name=name,
        default=default,
        fallbacks=tuple(fallbacks),
        params=dict(params or {}),
        choice=choice or ServerChoice(),
        requires=dict(requires or {}),
        description=description,
        backend=backend,
        capture=capture,
        retry=retry,
        daily_budget=daily_budget,
        settable=settable,
        embedding_backend=embedding_backend,
        adapt_input=adapt_input,
        model_params={k: dict(v or {}) for k, v in (model_params or {}).items()},
        media_backend=media_backend,
        fallback_on_refusal=fallback_on_refusal,
    )
    with _lock:
        _registry[name] = tp
    return tp


def get_touchpoint(name: str) -> Touchpoint:
    try:
        return _registry[name]
    except KeyError:
        raise UnknownTouchpoint(
            f"Unknown touchpoint {name!r}. Registered: {', '.join(sorted(_registry)) or 'none'}."
        ) from None


def all_touchpoints() -> list[Touchpoint]:
    return sorted(_registry.values(), key=lambda t: t.name)


def _unregister(name: str) -> None:
    """For tests."""
    _registry.pop(name, None)


__all__ = ["RetryPolicy", "Touchpoint", "all_touchpoints", "get_touchpoint", "touchpoint", "BaseChatModel"]
