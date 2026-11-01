"""Agent tools generated from Django models: declarations, scoping, writes, approvals, catalogue."""

import json
from datetime import date
from decimal import Decimal
from io import StringIO

import pytest
from django.contrib.auth.models import Permission
from django.core.exceptions import ImproperlyConfigured, PermissionDenied
from django.core.management import call_command
from rest_framework.test import APIClient

from llm_kit import tools
from llm_kit.core.runs import build_config
from llm_kit.models import ToolProposal
from llm_kit.tools import registry
from llm_kit.tools.caller import CONFIG_KEY, Caller
from llm_kit.tools.spec import AccessSpec
from tests.testapp.models import Customer, Invoice, Member, Note, Organisation, Product

pytestmark = pytest.mark.django_db


def grant(user, *codenames):
    user.user_permissions.add(*Permission.objects.filter(codename__in=codenames))
    return type(user).objects.get(pk=user.pk)  # fresh: has_perm caches


ALL = [f"{verb}_{m}" for verb in ("view", "add", "change") for m in ("organisation", "customer", "invoice", "note")] + [
    "view_product",
    "view_user",
]


@pytest.fixture
def world(django_user_model):
    acme, globex = Organisation.objects.create(name="Acme"), Organisation.objects.create(name="Globex")
    alice = django_user_model.objects.create_user("alice", first_name="Alice")
    bob = django_user_model.objects.create_user("bob")
    carol = django_user_model.objects.create_user("carol")  # in Acme, no permissions
    for u, org in ((alice, acme), (bob, globex), (carol, acme)):
        Member.objects.create(user=u, organisation=org)
    alice, bob = grant(alice, *ALL), grant(bob, *ALL)
    road = Customer.objects.create(organisation=acme, name="Roadrunner Ltd", email="beep@road.example")
    coyote = Customer.objects.create(organisation=acme, name="Coyote & Co", api_token="tok-SECRET")
    hidden = Customer.objects.create(organisation=globex, name="Globex Rival")
    Invoice.objects.create(organisation=acme, customer=road, number="A-1", total=10, issued_on=date(2026, 1, 5))
    Invoice.objects.create(
        organisation=acme, customer=coyote, number="A-2", total=20, status="sent", issued_on=date(2026, 2, 5)
    )
    Invoice.objects.create(organisation=globex, customer=hidden, number="G-1", total=99)
    Note.objects.create(owner=alice, text="call the roadrunner")
    Note.objects.create(owner=bob, text="bob's secret plan")
    Product.objects.create(name="Anvil", price=Decimal("9.99"))
    return {
        "acme": acme,
        "globex": globex,
        "alice": alice,
        "bob": bob,
        "carol": carol,
        "road": road,
        "coyote": coyote,
        "hidden": hidden,
    }


def tool(user, name):
    found = [t for t in tools.for_user(user) if t.name == name]
    assert found, f"{user} has no {name}"
    return found[0]


# ------------------------------------------------------------- declarations


@pytest.mark.parametrize(
    "options, error",
    [
        ({"fields": ["name"]}, "declare whose rows"),
        ({"fields": ["name"], "unscoped": ""}, "unscoped must say why"),
        ({"fields": ["name"], "unscoped": "x", "tenant_field": "organisation"}, "cannot be combined"),
        ({"fields": ["name", "api_token"], "tenant_field": "organisation"}, "looks like a secret"),
        ({"fields": [], "tenant_field": "organisation"}, "list the fields"),
        ({"fields": ["name"], "tenant_field": "organisation", "colour": "red"}, "unknown option"),
        ({"fields": ["name"], "tenant_field": "organisation", "permission": "view_customer"}, "app_label.codename"),
        (
            {"fields": ["name"], "tenant_field": "organisation", "writes": "none", "writable_fields": ["name"]},
            'writes = "none"',
        ),
        ({"fields": ["name"], "tenant_field": "organisation", "writable_fields": ["organisation"]}, "scope field"),
        ({"fields": ["name"], "owner_field": "organisation"}, "must lead to a link"),
        ({"fields": ["name"], "tenant_field": "organisation", "search_fields": ["organisation"]}, "is a link"),
    ],
)
def test_bad_customer_declarations_are_refused(options, error):
    with pytest.raises(ImproperlyConfigured, match=error):
        AccessSpec(Customer, options)


def test_write_rules():
    base = {"fields": ["number"], "tenant_field": "organisation"}
    with pytest.raises(ImproperlyConfigured, match="decision for a person"):
        AccessSpec(Invoice, {**base, "writable_fields": ["number", "customer", "status"]})
    with pytest.raises(ImproperlyConfigured, match="links to a user"):
        AccessSpec(Invoice, {**base, "writable_fields": ["number", "customer", "created_by"]})
    with pytest.raises(ImproperlyConfigured, match=r"required field\(s\) \['customer'\]"):
        AccessSpec(Invoice, {**base, "writable_fields": ["number"]})
    AccessSpec(Invoice, {**base, "writable_fields": ["number"], "creates": False})  # updates only: fine
    AccessSpec(Invoice, {**base, "writes": "approval", "writable_fields": ["number", "customer", "status"]})
    spec = AccessSpec(Note, {"fields": ["text"], "owner_field": "owner", "writable_fields": ["text"]})
    with pytest.raises(ImproperlyConfigured, match="which agents can't see"):
        AccessSpec(Invoice, {**base, "writable_fields": ["number", "customer"]}).check_links({Note: spec})


def test_duplicate_and_double_exposure_refused(world):
    with pytest.raises(ImproperlyConfigured, match="already declares AgentAccess"):
        tools.expose(Customer, fields=["name"], tenant_field="organisation")
    with pytest.raises(ImproperlyConfigured, match="two tools are named 'whoami'"):
        tools.register(registry._entries["whoami"].inner, permission=tools.AUTHENTICATED)


# ------------------------------------------------------------- catalogue


def test_catalogue_filtered_by_permission(world):
    names = {e.name for e in tools.catalogue()}
    assert {
        "find_customers",
        "save_customer",
        "find_invoices",
        "save_invoice",
        "find_notes",
        "save_note",
        "find_products",
        "find_organisations",
        "find_accounts",
        "whoami",
    } <= names
    assert "save_product" not in names and not any(n.startswith("delete") for n in names)
    assert {t.name for t in tools.for_user(world["alice"])} >= {"find_customers", "save_invoice", "whoami"}
    assert [t.name for t in tools.for_user(world["carol"])] == ["whoami"]
    assert tools.for_user(None) == []
    from django.contrib.auth.models import AnonymousUser

    assert tools.for_user(AnonymousUser()) == []
    assert [t.name for t in tools.for_user(world["alice"], kinds=["find"], categories=["auth"])] == ["find_accounts"]
    text = tools.guidance(tools.for_user(world["alice"], names=["find_customers", "whoami"]))
    assert "`find_customers`" in text and "`whoami` says who you are" in text


def test_schemas_are_small_and_typed(world):
    find = tool(world["alice"], "find_invoices").tool_call_schema.model_json_schema()["properties"]
    assert set(find) == {"id", "query", "limit", "offset", "status", "issued_on_from", "issued_on_to", "customer"}
    assert find["status"]["anyOf"][0]["enum"] == ["draft", "sent", "paid"]
    save = tool(world["alice"], "save_invoice").tool_call_schema.model_json_schema()["properties"]
    assert "organisation" not in save and "created_by" not in save
    assert "needs a person's approval" in save["status"]["description"]


# ------------------------------------------------------------- reading


def test_find_is_scoped_to_the_callers_tenant(world):
    find = tool(world["alice"], "find_customers")
    out = find.invoke({})
    assert "Roadrunner" in out and "Coyote" in out and "Globex" not in out
    assert "tok-SECRET" not in out  # api_token isn't listed (and couldn't be)
    assert find.invoke({"id": world["hidden"].pk}) == f"No customer has id {world['hidden'].pk}."
    assert "Globex Rival" in tool(world["bob"], "find_customers").invoke({})


def test_find_filters_search_pagination(world):
    find = tool(world["alice"], "find_invoices")
    assert "A-2" in find.invoke({"status": "sent"}) and "A-1" not in find.invoke({"status": "sent"})
    assert "A-1" in find.invoke({"query": "roadrunner"})  # across customer__name
    feb = find.invoke({"issued_on_from": "2026-02-01", "issued_on_to": "2026-02-28"})
    assert feb.startswith("1 match.") and "A-2" in feb
    page = find.invoke({"limit": 1})
    assert page.startswith("2 match; showing 1-1. Use offset=1 for more.") and "A-1" in page
    assert "A-2" in find.invoke({"limit": 1, "offset": 1})
    assert "status: sent (Sent)" in find.invoke({"id": Invoice.objects.get(number="A-2").pk})
    assert find.invoke({"issued_on_from": "not a date"}).startswith("Bad")


def test_owner_custom_and_unscoped(world):
    assert "roadrunner" in tool(world["alice"], "find_notes").invoke({})
    assert "bob's" not in tool(world["alice"], "find_notes").invoke({})
    accounts = tool(world["alice"], "find_accounts").invoke({})
    assert accounts.startswith("1 match.") and "alice" in accounts and "bob" not in accounts
    orgs = tool(world["alice"], "find_organisations").invoke({})
    assert "Acme" in orgs and "Globex" not in orgs
    assert "Anvil" in tool(world["bob"], "find_products").invoke({})


def test_no_tenant_means_no_rows_not_all_rows(world):
    find = tools.for_user(world["alice"], tenant=None, names=["find_customers"])[0]
    assert find.invoke({}) == "No customers match."


# ------------------------------------------------------------- writing


def test_direct_create_and_update(world):
    save = tool(world["alice"], "save_customer")
    out = save.invoke({"name": "Wile E."})
    assert out.startswith("Created customer.")
    created = Customer.objects.get(name="Wile E.")
    assert created.organisation == world["acme"]  # set by the server
    assert save.invoke({"id": created.pk, "email": "wile@acme.example"}).startswith("Changed customer.")
    created.refresh_from_db()
    assert created.email == "wile@acme.example" and created.name == "Wile E."  # untouched
    assert "email: Enter a valid email" in save.invoke({"id": created.pk, "email": "nope"})
    assert save.invoke({"id": world["hidden"].pk, "name": "x"}).startswith("No customer has id")
    assert save.invoke({"id": created.pk}).startswith("Nothing to change")


def test_links_are_limited_to_the_callers_rows(world):
    save = tool(world["alice"], "save_invoice")
    out = save.invoke({"number": "A-3", "customer": world["hidden"].pk})
    assert out.startswith("Not saved. customer: Select a valid choice")
    assert save.invoke({"number": "A-3", "customer": world["road"].pk}).startswith("Created invoice.")
    assert Invoice.objects.get(number="A-3").organisation == world["acme"]


def test_writes_that_leave_the_scope_are_rolled_back(world):
    spec = AccessSpec(
        Customer,
        {
            "fields": ["name"],
            "writable_fields": ["name"],
            "creates": False,
            "agent_queryset": lambda qs, caller: qs.filter(organisation_id=caller.tenant_id, name__startswith="R"),
        },
    )
    caller = Caller(world["alice"], world["acme"].pk)
    from llm_kit.tools import ops

    assert ops.save(spec, caller, id=world["road"].pk, name="Renamed").startswith("Changed")
    assert "outside the records you can see" in ops.save(spec, caller, id=world["road"].pk, name="Zebra")
    assert Customer.objects.get(pk=world["road"].pk).name == "Renamed"


def test_write_permission_checked_per_action(world, django_user_model):
    dave = django_user_model.objects.create_user("dave")
    Member.objects.create(user=dave, organisation=world["acme"])
    dave = grant(dave, "view_customer", "change_customer")  # may change, not add
    save = tool(dave, "save_customer")
    assert "permission to create" in save.invoke({"name": "New"})
    assert save.invoke({"id": world["road"].pk, "name": "Road2"}).startswith("Changed")


# ------------------------------------------------------------- approvals


def test_approval_field_makes_a_proposal(world):
    inv = Invoice.objects.get(number="A-1")
    out = tool(world["alice"], "save_invoice").invoke(
        {"id": inv.pk, "status": "paid", "notes": "thanks", "number": "A-1b"}
    )
    assert "Proposed #" in out and "must approve" in out
    inv.refresh_from_db()
    assert inv.status == "draft" and inv.notes == ""  # nothing changed yet, not even the direct field
    p = ToolProposal.objects.get()
    assert p.object_repr == "A-1"
    assert (p.action, p.before, p.after, p.tenant) == (
        "update",
        {"status": "draft", "notes": "", "number": "A-1"},
        {"status": "paid", "notes": "thanks", "number": "A-1b"},
        str(world["acme"].pk),
    )

    assert tools.pending_for(world["bob"]) == []  # other tenant
    with pytest.raises(PermissionDenied):
        tools.approve(p, by=world["bob"])
    with pytest.raises(PermissionDenied):
        tools.approve(p, by=world["carol"])  # same tenant, no permission
    assert tools.pending_for(world["alice"]) == [p]

    seen = []
    tools.proposal_decided.connect(lambda **kw: seen.append(kw["proposal"].status), weak=False, dispatch_uid="t")
    p = tools.approve(p, by=world["alice"])
    tools.proposal_decided.disconnect(dispatch_uid="t")
    assert p.status == "approved" and seen == ["approved"]
    inv.refresh_from_db()
    assert (inv.status, inv.notes, inv.number) == ("paid", "thanks", "A-1b")
    with pytest.raises(tools.ProposalError, match="already decided"):
        tools.approve(p, by=world["alice"])


def test_stale_proposal_fails_and_reject(world):
    inv = Invoice.objects.get(number="A-1")
    save = tool(world["alice"], "save_invoice")
    save.invoke({"id": inv.pk, "status": "sent"})
    Invoice.objects.filter(pk=inv.pk).update(status="paid")  # someone changed it meanwhile
    p = tools.approve(ToolProposal.objects.get(), by=world["alice"])
    assert p.status == "failed" and "changed after" in p.reason
    save.invoke({"id": inv.pk, "status": "draft"})
    p = tools.reject(ToolProposal.objects.get(status="pending"), by=world["alice"], reason="no")
    assert p.status == "rejected" and Invoice.objects.get(pk=inv.pk).status == "paid"


def test_approval_mode_create(world):
    out = tool(world["alice"], "save_note").invoke({"text": "buy anvils"})
    assert "Proposed #" in out and not Note.objects.filter(text="buy anvils").exists()
    p = tools.approve(ToolProposal.objects.get(), by=world["alice"])
    assert p.status == "approved" and Note.objects.get(pk=p.result_pk).owner == world["alice"]


# ------------------------------------------------------------- caller handling


def test_unbound_tools_need_a_server_set_caller(world):
    find = next(t for t in tools.all_tools() if t.name == "find_customers")
    assert find.invoke({}).startswith("Refused: no signed-in user")
    config = {"configurable": {CONFIG_KEY: Caller(world["carol"], world["acme"].pk)}}
    assert "don't have permission" in find.invoke({}, config)
    config = {"configurable": {CONFIG_KEY: Caller(world["alice"], world["acme"].pk)}}
    assert "Roadrunner" in find.invoke({}, config)
    # A client sending the key (it arrives as JSON, never as a Caller) gets nothing.
    assert find.invoke({}, {"configurable": {CONFIG_KEY: {"user": 1}}}).startswith("Refused")


def test_clients_cannot_set_dunder_configurable():
    config = build_config("t1", {"config": {"configurable": {CONFIG_KEY: "x", "__pregel_x": 1, "ok": 2}}})
    assert config["configurable"] == {"ok": 2, "thread_id": "t1"}


def test_hand_written_tool_sees_the_caller(world):
    assert tool(world["carol"], "whoami").invoke({}) == f"user=carol tenant={world['acme'].pk}"


@pytest.mark.django_db(transaction=True)
async def test_async_invocation(world):
    from asgiref.sync import sync_to_async

    find = (await sync_to_async(tools.for_user)(world["alice"], names=["find_customers"]))[0]
    assert "Roadrunner" in await find.ainvoke({})


# ------------------------------------------------------------- through a real run, API, command


@pytest.mark.django_db(transaction=True)
def test_graph_run_gets_the_caller_from_the_server(world):
    api = APIClient()
    api.force_authenticate(world["alice"])
    tid = api.post("/llm/api/threads/", {"graph": "tools"}, format="json").data["id"]
    ask = json.dumps({"tool": "find_customers", "args": {}})
    r = api.post(f"/llm/api/threads/{tid}/runs/", {"message": ask}, format="json")
    assert r.status_code == 200, r.data
    msgs = r.data["values"]["messages"]
    assert "find_customers" in msgs[1]["content"] and "find_products" in msgs[1]["content"]  # bound per caller
    assert "Roadrunner" in msgs[-1]["content"] and "Globex" not in msgs[-1]["content"]
    # The caller is not in the checkpoint, and a client can't smuggle one in.
    smuggle = {"message": ask, "config": {"configurable": {CONFIG_KEY: {"user": world["bob"].pk}}}}
    r = api.post(f"/llm/api/threads/{tid}/runs/", smuggle, format="json")
    assert "Globex" not in r.data["values"]["messages"][-1]["content"]


def test_api_catalogue_and_proposals(world):
    api = APIClient()
    assert api.get("/llm/api/tools/").status_code in (401, 403)
    api.force_authenticate(world["alice"])
    listed = {t["name"]: t for t in api.get("/llm/api/tools/").data}
    assert listed["save_invoice"]["writes"] == "direct" and listed["save_note"]["writes"] == "approval"
    assert "status" in listed["find_invoices"]["parameters"]["properties"]
    assert [t["name"] for t in api.get("/llm/api/tools/?kind=custom").data] == ["whoami"]

    tool(world["alice"], "save_note").invoke({"text": "via api"})
    pending = api.get("/llm/api/tools/proposals/").data
    assert len(pending) == 1 and pending[0]["after"] == {"text": "via api"}
    bob = APIClient()
    bob.force_authenticate(world["bob"])
    assert bob.post(f"/llm/api/tools/proposals/{pending[0]['id']}/approve/").status_code == 403
    r = api.post(f"/llm/api/tools/proposals/{pending[0]['id']}/approve/")
    assert r.status_code == 200 and r.data["status"] == "approved"
    assert api.post(f"/llm/api/tools/proposals/{pending[0]['id']}/reject/").status_code == 400


def test_admin_actions(world, client, django_user_model):
    tool(world["alice"], "save_note").invoke({"text": "admin approves"})
    admin_user = django_user_model.objects.create_superuser("root", "r@example.com", "pw")
    client.force_login(admin_user)
    assert client.get("/admin/llm_kit/toolproposal/").status_code == 200
    p = ToolProposal.objects.get()
    client.post("/admin/llm_kit/toolproposal/", {"action": "approve_selected", "_selected_action": [p.pk]})
    p.refresh_from_db()
    assert p.status == "approved" and p.decided_by == admin_user


def test_command(world):
    out = StringIO()
    call_command("llm_kit_tools", stdout=out)
    assert "find_invoices" in out.getvalue() and "scope: tenant=organisation" in out.getvalue()
    out = StringIO()
    call_command("llm_kit_tools", "--user", "carol", stdout=out)
    assert out.getvalue().startswith("1 tool(s) for carol")
