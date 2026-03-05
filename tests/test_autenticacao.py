from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from ip_mensageria_alocacao_api import routes
from ip_mensageria_alocacao_api.core import autenticacao


class MockResult:
    def __init__(self, rows):
        self.rows = rows
        self.total_rows = len(rows)
        self._iter = iter(rows)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


def test_verificar_senha():
    plain = "password123"
    hashed = autenticacao.obter_hash_senha(plain)
    assert autenticacao.verificar_senha(plain, hashed)
    assert not autenticacao.verificar_senha("wrong", hashed)


def test_obter_hash_senha():
    password = "testpassword"
    hashed = autenticacao.obter_hash_senha(password)
    assert isinstance(hashed, str)
    assert hashed != password


def test_obter_usuario_success(monkeypatch):
    mock_row = Mock()
    mock_row.usuario = "testuser"
    mock_row.senha_hash = "hashed"
    mock_row.desativado = False
    mock_result = MockResult([mock_row])
    mock_query_job = Mock()
    mock_query_job.result.return_value = mock_result
    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    monkeypatch.setattr(
        autenticacao,
        "make_bq_client",
        Mock(return_value=mock_client),
    )

    user = autenticacao.obter_usuario("testuser")
    assert user.usuario_nome == "testuser"
    assert user.senha_hash == "hashed"
    assert not user.desativado


def test_obter_usuario_not_found(monkeypatch):
    mock_result = MockResult([])
    mock_query_job = Mock()
    mock_query_job.result.return_value = mock_result
    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    monkeypatch.setattr(
        autenticacao,
        "make_bq_client",
        Mock(return_value=mock_client),
    )

    user = autenticacao.obter_usuario("nonexistent")
    assert user is None


def test_obter_usuario_none_username():
    user = autenticacao.obter_usuario(None)
    assert user is None


def test_autenticar_usuario_success(monkeypatch):
    mock_row = Mock()
    mock_row.usuario = "testuser"
    mock_row.senha_hash = autenticacao.obter_hash_senha("password")
    mock_row.desativado = False
    mock_result = MockResult([mock_row])
    mock_query_job = Mock()
    mock_query_job.result.return_value = mock_result
    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    monkeypatch.setattr(
        autenticacao,
        "make_bq_client",
        Mock(return_value=mock_client),
    )

    user = autenticacao.autenticar_usuario("testuser", "password")
    assert user.usuario_nome == "testuser"


def test_autenticar_usuario_wrong_password(monkeypatch):
    mock_row = Mock()
    mock_row.usuario = "testuser"
    mock_row.senha_hash = autenticacao.obter_hash_senha("password")
    mock_row.desativado = False
    mock_result = MockResult([mock_row])
    mock_query_job = Mock()
    mock_query_job.result.return_value = mock_result
    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    monkeypatch.setattr(
        autenticacao,
        "make_bq_client",
        Mock(return_value=mock_client),
    )

    user = autenticacao.autenticar_usuario("testuser", "wrong")
    assert user is False


def test_autenticar_usuario_nao_encontrado(monkeypatch):
    mock_result = MockResult([])
    mock_query_job = Mock()
    mock_query_job.result.return_value = mock_result
    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    monkeypatch.setattr(
        autenticacao,
        "make_bq_client",
        Mock(return_value=mock_client),
    )

    user = autenticacao.autenticar_usuario("nonexistent", "password")
    assert user is False


def test_criar_token_acesso():
    data = {"sub": "testuser"}
    token = autenticacao.criar_token_acesso(data)
    assert isinstance(token, str)
    assert len(token) > 0


def test_criar_token_acesso_com_expiracao():
    data = {"sub": "testuser"}
    expires = timedelta(minutes=30)
    token = autenticacao.criar_token_acesso(data, expires)
    assert isinstance(token, str)


@pytest.mark.asyncio
async def test_login_para_token_successo(monkeypatch):
    # Mock the query
    mock_row = Mock()
    mock_row.usuario = "testuser"
    mock_row.senha_hash = autenticacao.obter_hash_senha("password")
    mock_row.desativado = False
    mock_result = MockResult([mock_row])
    mock_query_job = Mock()
    mock_query_job.result.return_value = mock_result
    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    monkeypatch.setattr(
        autenticacao,
        "make_bq_client",
        Mock(return_value=mock_client),
    )

    from fastapi.security import OAuth2PasswordRequestForm

    form_data = OAuth2PasswordRequestForm(username="testuser", password="password")

    response = await routes.login_para_token(form_data)
    assert "access_token" in response
    assert response["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_para_token_falha(monkeypatch):
    # Mock the query to return no user
    mock_result = MockResult([])
    mock_query_job = Mock()
    mock_query_job.result.return_value = mock_result
    mock_client = Mock()
    mock_client.query.return_value = mock_query_job
    monkeypatch.setattr(
        autenticacao,
        "make_bq_client",
        Mock(return_value=mock_client),
    )

    from fastapi.security import OAuth2PasswordRequestForm

    form_data = OAuth2PasswordRequestForm(username="testuser", password="password")

    with pytest.raises(HTTPException) as exc_info:
        await routes.login_para_token(form_data)
    assert exc_info.value.status_code == 401


# Test authentication edge cases


def test_obter_usuario_atual_via_api_key_missing_header():
    """Test obter_usuario_atual_via_api_key with missing X-Api-Key header."""
    with pytest.raises(HTTPException) as exc_info:
        autenticacao.obter_usuario_atual_via_api_key(None)
    assert exc_info.value.status_code == 400
    assert "No api key provided" in str(exc_info.value.detail)


@patch("ip_mensageria_alocacao_api.core.autenticacao.jwt.decode")
def test_obter_usuario_atual_via_api_key_invalid_jwt(mock_jwt_decode):
    """Test obter_usuario_atual_via_api_key with invalid JWT."""
    from jose import JWTError

    mock_jwt_decode.side_effect = JWTError("Invalid JWT")

    with pytest.raises(HTTPException) as exc_info:
        autenticacao.obter_usuario_atual_via_api_key("invalid_token")
    assert exc_info.value.status_code == 401


@patch("ip_mensageria_alocacao_api.core.autenticacao.jwt.decode")
@patch("ip_mensageria_alocacao_api.core.autenticacao.obter_usuario")
def test_obter_usuario_atual_via_api_key_user_not_found(
    mock_obter_usuario, mock_jwt_decode
):
    """Test obter_usuario_atual_via_api_key when user doesn't exist."""
    mock_jwt_decode.return_value = {"sub": "nonexistent"}
    mock_obter_usuario.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        autenticacao.obter_usuario_atual_via_api_key("valid_token")
    assert exc_info.value.status_code == 401


@patch("ip_mensageria_alocacao_api.core.autenticacao.jwt.decode")
def test_obter_usuario_atual_via_api_key_missing_sub(mock_jwt_decode):
    """Test obter_usuario_atual_via_api_key with JWT missing 'sub' claim."""
    mock_jwt_decode.return_value = {}

    with pytest.raises(HTTPException) as exc_info:
        autenticacao.obter_usuario_atual_via_api_key("token_without_sub")
    assert exc_info.value.status_code == 401
