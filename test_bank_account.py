import pytest

from bank_account import BankAccount


def test_initial_balance_defaults_to_zero():
    assert BankAccount().balance == 0


def test_initial_balance_can_be_set():
    assert BankAccount(100).balance == 100


def test_deposit_increases_balance():
    account = BankAccount(50)
    account.deposit(25)
    assert account.balance == 75


def test_withdraw_decreases_balance():
    account = BankAccount(50)
    account.withdraw(20)
    assert account.balance == 30


@pytest.mark.parametrize("amount", [0, -1, -100])
def test_deposit_rejects_non_positive(amount):
    account = BankAccount(50)
    with pytest.raises(ValueError):
        account.deposit(amount)
    assert account.balance == 50


@pytest.mark.parametrize("amount", [0, -1, -100])
def test_withdraw_rejects_non_positive(amount):
    account = BankAccount(50)
    with pytest.raises(ValueError):
        account.withdraw(amount)
    assert account.balance == 50


def test_withdraw_insufficient_funds():
    account = BankAccount(50)
    with pytest.raises(ValueError, match="Insufficient funds"):
        account.withdraw(51)
    assert account.balance == 50
