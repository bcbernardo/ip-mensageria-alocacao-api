from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from google.cloud.bigquery.table import RowIterator, _EmptyRowIterator

from ip_mensageria_alocacao_api.core import auxiliar, modelos


class MockResult:
    def __init__(self, rows):
        self.rows = rows
        self.total_rows = len(rows)
        self._iter = iter(rows)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)


class DummyImputer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # simple numeric imputation: replace NaN with 0
        return df.fillna(0)


class DummyClassificador:
    def __init__(
        self, atributos_colunas, atributos_categoricos, template_dims=3, midia_dims=2
    ):
        self.atributos_colunas = atributos_colunas
        self.atributos_categoricos = atributos_categoricos
        self.imputador_numerico = DummyImputer()
        self.template_embedding_dims = template_dims
        self.midia_embedding_dims = midia_dims
        # used by converter_df_em_pool
        self.cat_feature_names = atributos_categoricos
        self.feature_cols = atributos_categoricos


def test_beta_from_mean_se_fallback():
    p = 0.5
    se = 1.0  # large se -> denom <= 0 -> fallback
    alpha, beta = auxiliar.beta_from_mean_se(p, se)
    assert alpha == pytest.approx(1.0 + 9.0 * p)
    assert beta == pytest.approx(1.0 + 9.0 * (1 - p))


def test_beta_from_mean_se_consistency():
    p = 0.2
    se = 0.05
    alpha, beta = auxiliar.beta_from_mean_se(p, se)
    # posterior mean should be close to p
    mean = alpha / (alpha + beta)
    assert mean == pytest.approx(p, rel=1e-3)
    assert alpha > 0
    assert beta > 0


def test_thompson_sample_range_and_distribution():
    p = 0.3
    se = 0.1
    samples = [auxiliar.thompson_sample(p, se) for _ in range(100)]
    assert all(0.0 <= s <= 1.0 for s in samples)
    # variability: not all identical
    assert not all(s == samples[0] for s in samples)


def test_preparar_atributos_para_predicao_basic():
    # prepare dummy artefato with expected columns
    template_dims = 3
    midia_dims = 2
    base_cols = [
        "linha_cuidado",
        "cidadao_sexo",
        "cidadao_raca_cor",
        "mensagem_dia_semana",
        "municipio_prop_domicilios_zona_rural",
        "cidadao_plano_saude_privado",
        "cidadao_idade",
        "cidadao_tempo_desde_ultimo_procedimento",
        "mensagem_horario_relativo_12h",
        "mensagem_tipo",
    ]
    # add embedding column names
    for i in range(template_dims):
        base_cols.append(f"template_emb_{i}")
    for i in range(midia_dims):
        base_cols.append(f"midia_emb_{i}")

    classificadores = DummyClassificador(
        base_cols, ["linha_cuidado", "cidadao_sexo"], template_dims, midia_dims
    )

    cidadao = modelos.CidadaoCaracteristicas(
        idade=30,
        plano_saude_privado=True,
        raca_cor="Parda",
        sexo="Feminino",
        tempo_desde_ultimo_procedimento=120,
        municipio_prop_domicilios_zona_rural=0.12,
    )

    mensagem_template_embedding = [0.1] * template_dims
    mensagem_midia_embedding = [0.2] * midia_dims

    df = auxiliar.preparar_atributos_para_predicao(
        classificadores=classificadores,
        cidadao_caracteristicas=cidadao,
        linha_cuidado=modelos.LinhaCuidado.citotopatologico,
        tempo_desde_ultimo_procedimento=None,
        mensagem_tipo=modelos.MensagemTipo.mensagem_inicial,
        mensagem_dia_semana=modelos.DiaSemana.segunda,
        mensagem_horario=15,
        mensagem_template_embedding=mensagem_template_embedding,
        mensagem_midia_embedding=mensagem_midia_embedding,
    )

    # dataframe should have the columns in classificadores.atributos_colunas
    assert list(df.columns) == classificadores.atributos_colunas
    # check some values
    assert df.loc[0, "mensagem_horario_relativo_12h"] == 3
    assert df.loc[0, "cidadao_plano_saude_privado"] in (0.0, 1.0)


def test_converter_df_em_pool_uses_cat_indices(monkeypatch: pytest.MonkeyPatch):
    # build artefato where 'catcol' is categorical and present in feature_cols
    atributos_colunas = ["num1", "catcol", "num2"]
    artefato = DummyClassificador(
        atributos_colunas, ["catcol"], template_dims=0, midia_dims=0
    )

    df = pd.DataFrame({"num1": [1.0], "catcol": ["A"], "num2": [2.0]})

    captured = {}

    def fake_Pool(df_arg, cat_features=None):
        captured["df"] = df_arg
        captured["cat_features"] = cat_features
        return "POOL_OBJ"

    monkeypatch.setattr(auxiliar, "Pool", fake_Pool)

    res = auxiliar.converter_df_em_pool(df, artefato)
    assert res == "POOL_OBJ"
    # catcol index should be 1
    assert captured["cat_features"] == [atributos_colunas.index("catcol")]


@pytest.fixture
def mock_query_result():
    class MockRow:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MockQueryResult:
        def __init__(self, rows, total_rows=None):
            self.rows = rows
            self.total_rows = total_rows or len(rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    return MockQueryResult


@patch("ip_mensageria_alocacao_api.core.auxiliar.make_bq_client")
def test_obter_caracteristicas_usuario(mock_make_bq_client):
    mock_row = Mock(
        idade=30,
        sexo="Feminino",
        raca_cor="Parda",
        plano_saude_privado=True,
        prop_domicilios_zona_rural=0.12,
    )
    mock_result = MockResult([mock_row])
    mock_query = Mock()
    mock_query.result.return_value = mock_result
    mock_client = Mock()
    mock_client.query.return_value = mock_query
    mock_make_bq_client.return_value = mock_client

    result = auxiliar.obter_caracteristicas_usuario("123")

    assert isinstance(result, modelos.CidadaoCaracteristicas)
    assert result.idade == 30
    assert result.sexo == "Feminino"
    assert result.raca_cor == "Parda"
    assert result.plano_saude_privado
    assert result.municipio_prop_domicilios_zona_rural == 0.12


@patch("ip_mensageria_alocacao_api.core.auxiliar.make_bq_client")
def test_obter_tempo_desde_ultimo_procedimento_citotopatologico(mock_make_bq_client):
    mock_row = Mock(tempo_desde_ultimo_procedimento=10)
    mock_result = MockResult([mock_row])
    mock_query = Mock()
    mock_query.result.return_value = mock_result
    mock_client = Mock()
    mock_client.query.return_value = mock_query
    mock_make_bq_client.return_value = mock_client

    result = auxiliar.obter_tempo_desde_ultimo_procedimento(
        "123", modelos.LinhaCuidado.citotopatologico
    )

    assert result == 10


@patch("ip_mensageria_alocacao_api.core.auxiliar.make_bq_client")
def test_obter_tempo_desde_ultimo_procedimento_cronicos(mock_make_bq_client):
    mock_result_diabetes = MockResult([Mock(tempo_desde_ultimo_procedimento=20)])
    mock_result_hipertensao = MockResult([Mock(tempo_desde_ultimo_procedimento=15)])

    mock_query_diabetes = Mock()
    mock_query_diabetes.result.return_value = mock_result_diabetes

    mock_query_hipertensao = Mock()
    mock_query_hipertensao.result.return_value = mock_result_hipertensao

    mock_client = Mock()
    mock_client.query.side_effect = [mock_query_diabetes, mock_query_hipertensao]
    mock_make_bq_client.return_value = mock_client

    # Need to patch isinstance to recognize MockResult as RowIterator
    with patch(
        "ip_mensageria_alocacao_api.core.auxiliar.isinstance"
    ) as mock_isinstance:

        def isinstance_side_effect(obj, cls):
            if cls is RowIterator and isinstance(obj, MockResult):
                return True
            return isinstance.__wrapped__(obj, cls)

        mock_isinstance.side_effect = isinstance_side_effect

        result = auxiliar.obter_tempo_desde_ultimo_procedimento(
            "123", modelos.LinhaCuidado.cronicos
        )

    assert result == 15  # min of 20 and 15


@patch("ip_mensageria_alocacao_api.core.auxiliar.make_bq_client")
def test_obter_template_embedding_por_nome_success(mock_make_bq_client):
    mock_row = Mock(embedding=[0.1, 0.2, 0.3])
    mock_result = MockResult([mock_row])
    mock_query = Mock()
    mock_query.result.return_value = mock_result
    mock_client = Mock()
    mock_client.query.return_value = mock_query
    mock_make_bq_client.return_value = mock_client

    result = auxiliar.obter_template_embedding_por_nome("template1")

    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array([0.1, 0.2, 0.3]))


@patch("ip_mensageria_alocacao_api.core.auxiliar.make_bq_client")
def test_obter_template_embedding_por_nome_not_found(mock_make_bq_client):
    mock_query = Mock()
    mock_query.result.return_value = _EmptyRowIterator()
    mock_client = Mock()
    mock_client.query.return_value = mock_query
    mock_make_bq_client.return_value = mock_client

    with pytest.raises(HTTPException):
        auxiliar.obter_template_embedding_por_nome("nonexistent")


@patch("ip_mensageria_alocacao_api.core.auxiliar.make_bq_client")
def test_obter_template_embedding_por_texto_cached(mock_make_bq_client):
    # First query returns empty result
    mock_empty = _EmptyRowIterator()
    # Second query returns result with embedding
    mock_row = Mock(embedding=[0.4, 0.5, 0.6])
    mock_result = MockResult([mock_row])
    mock_query_with_result = Mock()
    mock_query_with_result.result.return_value = mock_result

    # Set up the mock to return empty result on first call, then query with result on second call
    mock_client = Mock()
    mock_client.query.side_effect = [mock_empty, mock_query_with_result]
    mock_make_bq_client.return_value = mock_client

    result = auxiliar.obter_template_embedding_por_texto("text", "btn0", "btn1", "btn2")

    assert isinstance(result, np.ndarray)


@patch("ip_mensageria_alocacao_api.core.auxiliar.make_bq_client")
def test_obter_midia_embedding_gs(mock_make_bq_client):
    mock_row = Mock(embedding=[0.7, 0.8])
    mock_result = MockResult([mock_row])
    mock_query = Mock()
    mock_query.result.return_value = mock_result
    mock_client = Mock()
    mock_client.query.return_value = mock_query
    mock_make_bq_client.return_value = mock_client

    result = auxiliar.obter_midia_embedding("gs://bucket/file.jpg")

    assert isinstance(result, np.ndarray)


@patch("ip_mensageria_alocacao_api.core.auxiliar.make_bq_client")
def test_obter_midia_embedding_http(mock_make_bq_client):
    mock_row = Mock(embedding=[0.9, 1.0])
    mock_result = MockResult([mock_row])
    mock_query = Mock()
    mock_query.result.return_value = mock_result
    mock_client = Mock()
    mock_client.query.return_value = mock_query
    mock_make_bq_client.return_value = mock_client

    result = auxiliar.obter_midia_embedding("http://example.com/image.jpg")

    assert isinstance(result, np.ndarray)


@patch("ip_mensageria_alocacao_api.core.auxiliar.make_bq_client")
def test_obter_midia_embedding_invalid_url(mock_make_bq_client):
    with pytest.raises(HTTPException):  # HTTPException
        auxiliar.obter_midia_embedding("ftp://invalid.com/file.jpg")


def test_preparar_atributos_para_predicao_missing_cidadao_data():
    """Test preparar_atributos_para_predicao with None values in CidadaoCaracteristicas."""
    from ip_mensageria_alocacao_api.core import auxiliar, modelos

    # Create classificadores mock
    classificadores = Mock()
    classificadores.atributos_colunas = [
        "linha_cuidado",
        "cidadao_sexo",
        "cidadao_raca_cor",
        "mensagem_dia_semana",
        "municipio_prop_domicilios_zona_rural",
        "cidadao_plano_saude_privado",
        "cidadao_idade",
        "cidadao_tempo_desde_ultimo_procedimento",
        "mensagem_horario_relativo_12h",
        "mensagem_tipo",
    ]
    classificadores.atributos_categoricos = ["linha_cuidado", "cidadao_sexo"]
    classificadores.imputador_numerico = Mock()
    classificadores.imputador_numerico.transform.return_value = [[0.1, 1, 25, 10, 3]]

    # CidadaoCaracteristicas with None values
    cidadao = modelos.CidadaoCaracteristicas(
        idade=None,
        plano_saude_privado=None,
        raca_cor=None,
        sexo=None,
        tempo_desde_ultimo_procedimento=None,
        municipio_prop_domicilios_zona_rural=None,
    )

    df = auxiliar.preparar_atributos_para_predicao(
        classificadores=classificadores,
        cidadao_caracteristicas=cidadao,
        linha_cuidado=modelos.LinhaCuidado.citotopatologico,
        tempo_desde_ultimo_procedimento=10,
        mensagem_tipo=modelos.MensagemTipo.mensagem_inicial,
        mensagem_dia_semana=modelos.DiaSemana.segunda,
        mensagem_horario=15,
        mensagem_template_embedding=[],
        mensagem_midia_embedding=[],
    )

    # Should handle None values gracefully (imputation should work)
    assert df is not None
    assert len(df) == 1
    # Check that MISSING was filled for categorical None values
    assert df.loc[0, "cidadao_sexo"] == "MISSING"
    assert df.loc[0, "cidadao_raca_cor"] == "MISSING"


def test_beta_from_mean_se_edge_cases():
    """Test beta_from_mean_se with edge case inputs."""
    from ip_mensageria_alocacao_api.core import auxiliar

    # Test with p = 0
    alpha, beta = auxiliar.beta_from_mean_se(0.0, 0.1)
    assert alpha > 0
    assert beta > 0

    # Test with p = 1
    alpha, beta = auxiliar.beta_from_mean_se(1.0, 0.1)
    assert alpha > 0
    assert beta > 0

    # Test with very small se
    alpha, beta = auxiliar.beta_from_mean_se(0.5, 1e-10)
    assert alpha > 0
    assert beta > 0

    # Test with very large se (should trigger fallback)
    alpha, beta = auxiliar.beta_from_mean_se(0.5, 10.0)
    assert alpha == 1.0 + 9.0 * 0.5  # fallback formula
    assert beta == 1.0 + 9.0 * 0.5
