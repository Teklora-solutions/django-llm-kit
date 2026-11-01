from tests.testapp.models import Member


def tenant_for(user):
    return Member.objects.filter(user=user).values_list("organisation_id", flat=True).first()
