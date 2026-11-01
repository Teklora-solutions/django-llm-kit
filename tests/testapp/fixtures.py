"""A small world for tool and agent tests: two organisations, their users, customers and invoices."""

from datetime import date
from decimal import Decimal

import pytest
from django.contrib.auth.models import Permission

from tests.testapp.models import Customer, Invoice, Member, Note, Organisation, Product


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
