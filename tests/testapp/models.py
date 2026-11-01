"""Models for the model-tools tests: tenant-, owner-, custom- and un-scoped."""

from django.conf import settings
from django.db import models


class Organisation(models.Model):
    name = models.CharField(max_length=100)

    class AgentAccess:
        description = "the organisation you work for"
        fields = ["name"]

        def agent_queryset(qs, caller):
            return qs.filter(pk=caller.tenant_id)

    def __str__(self):
        return self.name


class Member(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    organisation = models.ForeignKey(Organisation, on_delete=models.CASCADE)


class Customer(models.Model):
    organisation = models.ForeignKey(Organisation, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    email = models.EmailField(blank=True)
    api_token = models.CharField(max_length=64, blank=True)

    class AgentAccess:
        description = "companies that buy from us"
        fields = ["name", "email"]
        search_fields = ["name", "email"]
        tenant_field = "organisation"
        writable_fields = ["name", "email"]

    def __str__(self):
        return self.name


class Invoice(models.Model):
    STATUSES = [("draft", "Draft"), ("sent", "Sent"), ("paid", "Paid")]
    organisation = models.ForeignKey(Organisation, on_delete=models.CASCADE)
    customer = models.ForeignKey(Customer, on_delete=models.PROTECT)
    number = models.CharField(max_length=20)
    total = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    status = models.CharField(max_length=10, choices=STATUSES, default="draft")
    issued_on = models.DateField(null=True, blank=True)
    notes = models.TextField(blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)

    class Meta:
        ordering = ["number"]

    class AgentAccess:
        description = "invoices we sent to customers"
        fields = ["number", "customer", "total", "status", "issued_on", "notes"]
        search_fields = ["number", "customer__name", "notes"]
        filter_fields = ["status", "issued_on", "customer"]
        tenant_field = "organisation"
        writes = "direct"
        writable_fields = ["number", "customer", "total", "issued_on", "notes"]
        approval_fields = ["status"]

    def __str__(self):
        return self.number


class Note(models.Model):
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    text = models.TextField()
    pinned = models.BooleanField(default=False)

    class AgentAccess:
        description = "the user's private notes"
        fields = ["text", "pinned"]
        search_fields = ["text"]
        filter_fields = ["pinned"]
        owner_field = "owner"
        writes = "approval"
        writable_fields = ["text", "pinned"]


class Product(models.Model):
    name = models.CharField(max_length=100)
    price = models.DecimalField(max_digits=10, decimal_places=2)

    class AgentAccess:
        description = "our public price list"
        fields = ["name", "price"]
        search_fields = ["name"]
        unscoped = "the price list is public"
