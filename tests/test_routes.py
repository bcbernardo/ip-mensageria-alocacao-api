from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from ip_mensageria_alocacao_api.core.autenticacao import obter_usuario_atual_via_api_key
from ip_mensageria_alocacao_api.core.modelos import UsuarioNaBase
from ip_mensageria_alocacao_api.main import create_app


@pytest.fixture
def client():
    """Create test client with mock classificadores."""
    app = create_app(carregar_classificadores_na_inicializacao=False)
    # Mock classificadores to avoid GCS dependency
    app.state.classificadores = Mock()
    app.state.classificadores.modelos = [Mock()]
    app.state.classificadores.template_embedding_dims = 3
    app.state.classificadores.midia_embedding_dims = 2
    app.state.classificadores.atributos_colunas = ["col1", "col2"]
    app.state.classificadores.atributos_categoricos = []
    app.state.classificadores.modelos[0].predict_proba.return_value = [[0.3, 0.7]]
    return TestClient(app)


def test_index_endpoint(client):
    """Test the index endpoint returns correct response."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "info" in data
    assert "API" in data["info"]


@patch("ip_mensageria_alocacao_api.routes.autenticar_usuario")
def test_login_success(mock_auth, client):
    """Test successful login returns token."""
    mock_user = UsuarioNaBase(
        usuario_nome="testuser", senha_hash="hash", desativado=False
    )
    mock_auth.return_value = mock_user

    response = client.post("/token", data={"username": "testuser", "password": "pass"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "token_type" in data
    assert data["token_type"] == "bearer"


@patch("ip_mensageria_alocacao_api.routes.autenticar_usuario")
def test_login_failure(mock_auth, client):
    """Test login failure returns 401."""
    mock_auth.return_value = False

    response = client.post("/token", data={"username": "testuser", "password": "wrong"})
    assert response.status_code == 401
    assert "WWW-Authenticate" in response.headers


def test_prever_efetividade_missing_auth(client):
    """Test prediction endpoint requires authentication."""
    response = client.post("/prever_efetividade_mensagem", json={})
    assert (
        response.status_code == 400
    )  # FastAPI returns 400 for missing required fields


def test_alocar_missing_auth(client):
    """Test allocation endpoint requires authentication."""
    response = client.post("/alocar", json=[])
    assert response.status_code == 400


def test_prever_efetividade_missing_classificadores():
    """Test prediction returns 503 when lazy classifier load fails."""
    app = create_app(carregar_classificadores_na_inicializacao=False)
    app.dependency_overrides[obter_usuario_atual_via_api_key] = lambda: UsuarioNaBase(
        usuario_nome="testuser", senha_hash="hash", desativado=False
    )

    client = TestClient(app)
    with patch(
        "ip_mensageria_alocacao_api.routes.carregar_classificadores",
        side_effect=RuntimeError("credenciais ausentes"),
    ):
        response = client.post(
            "/prever_efetividade_mensagem",
            params={
                "cidadao_id": "123",
                "linha_cuidado": "crônicos",
                "mensagem_tipo": "mensagem_inicial",
            },
            json={"dia_semana": "Monday", "horario": 10},
            headers={"X-Api-Key": "fake"},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "credenciais ausentes"
