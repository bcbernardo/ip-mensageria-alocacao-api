from __future__ import annotations

from functools import lru_cache
from http import HTTPStatus
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import Pool
from fastapi import HTTPException
from google.cloud.bigquery.table import RowIterator, _EmptyRowIterator
from numpy import dtype, ndarray
from pydantic import AnyUrl

from ip_mensageria_alocacao_api.core.bd import make_bq_client
from ip_mensageria_alocacao_api.core.configs import BQ_PROJETO
from ip_mensageria_alocacao_api.core.modelos import (
    CidadaoCaracteristicas,
    Classificador,
    DiaSemana,
    LinhaCuidado,
    MensagemTipo,
)


def beta_from_mean_se(p: float, se: float, eps: float = 1e-6) -> Tuple[float, float]:
    """
    Aproxima Beta(α,β) a partir de média p e desvio padrão se.
    Se a variância for muito pequena, aplica 'floor' para evitar α,β → ∞.
    """
    p = min(max(p, eps), 1 - eps)
    v = max(se**2, 1e-5)  # piso de variância conservador
    # var_beta = p(1-p)/(α+β+1)  => α+β = p(1-p)/v - 1
    denom = (p * (1 - p) / v) - 1.0
    if denom <= 0:
        # fallback: algo difuso porém informativo
        return 1.0 + 9.0 * p, 1.0 + 9.0 * (1 - p)
    alpha = p * denom
    beta = (1 - p) * denom
    # evita degeneração
    alpha = float(max(alpha, eps))
    beta = float(max(beta, eps))
    return alpha, beta


@lru_cache(maxsize=128)
def obter_caracteristicas_usuario(cidadao_id: str) -> CidadaoCaracteristicas:
    query = f"""
        SELECT
            DATE_DIFF(
                CURRENT_DATE(),
                SAFE_CAST(c.cidadao_dt_nascimento AS DATE),
                YEAR
            ) AS idade,
            c.cidadao_sexo as sexo,
            c.cidadao_raca_cor as raca_cor,
            c.cidadao_plano_saude_privado as plano_saude_privado,
            m.perc_dom_zona_rural as prop_domicilios_zona_rural
        FROM `ip_mensageria_camada_ouro.cidadao` c
        LEFT JOIN `pmai_camada_prata.situacao_domicilios_municipios_censo_2010` m
        ON c.municipio_id_sus = m.cod_mun_ibge
        WHERE c.id = '{cidadao_id}'
    """
    resultado_query = make_bq_client().query(query).result()
    assert resultado_query.total_rows == 1
    cidadao = next(resultado_query)
    return CidadaoCaracteristicas(
        idade=cidadao.idade,
        plano_saude_privado=cidadao.plano_saude_privado,
        raca_cor=cidadao.raca_cor,
        sexo=cidadao.sexo,
        municipio_prop_domicilios_zona_rural=cidadao.prop_domicilios_zona_rural,
        tempo_desde_ultimo_procedimento=None,
    )


@lru_cache(maxsize=128)
def obter_tempo_desde_ultimo_procedimento(
    cidadao_id: str,
    linha_cuidado: LinhaCuidado,
) -> int:
    query_cito = f"""
        SELECT
            MIN(DATE_DIFF(
                CURRENT_DATE(),
                dt_ultimo_exame,
                DAY
            )) AS tempo_desde_ultimo_procedimento
        FROM `ip_camada_prata_historico_transmissoes.previne_brasil_citopatologico_mensageria`
        WHERE cidadao_id = '{cidadao_id}'
    """

    query_diabetes = f"""
        SELECT
            MIN(DATE_DIFF(
                CURRENT_DATE(),
                GREATEST(
                    dt_solicitacao_hemoglobina_glicada_mais_recente,
                    dt_consulta_mais_recente
                ),
                DAY
            )) AS tempo_desde_ultimo_procedimento
        FROM `ip_camada_prata_historico_transmissoes.previne_brasil_diabeticos_mensageria`
        WHERE cidadao_id = '{cidadao_id}'
    """

    query_hipertensao = f"""
        SELECT
            MIN(DATE_DIFF(
                CURRENT_DATE(),
                GREATEST(
                    dt_afericao_pressao_mais_recente,
                    dt_consulta_mais_recente
                ),
                DAY
            )) AS tempo_desde_ultimo_procedimento
        FROM `ip_camada_prata_historico_transmissoes.previne_brasil_hipertensos_mensageria`
        WHERE cidadao_id = '{cidadao_id}'
    """

    if linha_cuidado == LinhaCuidado.citotopatologico:
        resultado_query = make_bq_client().query(query_cito).result()
        assert resultado_query.total_rows == 1
        tempo_desde_ultimo_procedimento = next(
            resultado_query
        ).tempo_desde_ultimo_procedimento
    elif linha_cuidado == LinhaCuidado.cronicos:
        resultado_query_diabetes = make_bq_client().query(query_diabetes).result()
        resultado_query_hipertensao = make_bq_client().query(query_hipertensao).result()
        if isinstance(resultado_query_diabetes, RowIterator) and isinstance(
            resultado_query_hipertensao, RowIterator
        ):
            tempo_proc_dia = next(
                resultado_query_diabetes
            ).tempo_desde_ultimo_procedimento
            tempo_proc_hiper = next(
                resultado_query_hipertensao
            ).tempo_desde_ultimo_procedimento
            tempo_desde_ultimo_procedimento = None
            if tempo_proc_dia is not None and tempo_proc_hiper is not None:
                tempo_desde_ultimo_procedimento = np.minimum(
                    tempo_proc_dia or np.inf,
                    tempo_proc_hiper or np.inf,
                )
        elif isinstance(resultado_query_diabetes, RowIterator):
            tempo_desde_ultimo_procedimento = next(
                resultado_query_diabetes
            ).tempo_desde_ultimo_procedimento
        elif isinstance(resultado_query_hipertensao, RowIterator):
            tempo_desde_ultimo_procedimento = next(
                resultado_query_hipertensao
            ).tempo_desde_ultimo_procedimento
        else:
            raise ValueError(
                f"Cidadão não encontrado nas listas de {linha_cuidado.value}."
            )
    else:
        raise ValueError(f"Linha de cuidado {linha_cuidado} não suportada.")

    return tempo_desde_ultimo_procedimento


def preparar_atributos_para_predicao(
    *,
    classificadores: Classificador,
    cidadao_caracteristicas: CidadaoCaracteristicas,
    linha_cuidado: LinhaCuidado,
    tempo_desde_ultimo_procedimento: Optional[int],
    mensagem_tipo: MensagemTipo,
    mensagem_dia_semana: DiaSemana,
    mensagem_horario: int,
    mensagem_template_embedding: list[float] | ndarray[Any, dtype[Any]],
    mensagem_midia_embedding: list[float] | ndarray[Any, dtype[Any]],
) -> pd.DataFrame:
    # categóricas do treino: linha_cuidado, cidadao_sexo, cidadao_raca_cor, mensagem_dia_semana
    # numéricas do treino: municipio_prop..., plano_privado, idade, tempo_desde..., mensagem_horario_relativo_12h
    row = {
        "linha_cuidado": str(linha_cuidado.value),
        "cidadao_sexo": (cidadao_caracteristicas.sexo or "MISSING"),
        "cidadao_raca_cor": (cidadao_caracteristicas.raca_cor or "MISSING"),
        "mensagem_dia_semana": str(mensagem_dia_semana.value),
        "municipio_prop_domicilios_zona_rural": cidadao_caracteristicas.municipio_prop_domicilios_zona_rural,
        "cidadao_plano_saude_privado": int(
            bool(cidadao_caracteristicas.plano_saude_privado)
        )
        if cidadao_caracteristicas.plano_saude_privado is not None
        else None,
        "cidadao_idade": cidadao_caracteristicas.idade,
        "cidadao_tempo_desde_ultimo_procedimento": tempo_desde_ultimo_procedimento
        if tempo_desde_ultimo_procedimento is not None
        else cidadao_caracteristicas.tempo_desde_ultimo_procedimento,
        "mensagem_horario_relativo_12h": int(mensagem_horario) - 12,
        "mensagem_tipo": str(mensagem_tipo.value),
    }

    # embeddings
    for i, v in enumerate(mensagem_template_embedding):
        row[f"template_emb_{i}"] = float(v)
    for i, v in enumerate(mensagem_midia_embedding):
        row[f"midia_emb_{i}"] = float(v)

    # DataFrame de 1 linha com TODAS as colunas esperadas pelo modelo
    df = pd.DataFrame([row], columns=None)

    # imputação numérica como no treino
    num_cols = [
        "municipio_prop_domicilios_zona_rural",
        "cidadao_plano_saude_privado",
        "cidadao_idade",
        "cidadao_tempo_desde_ultimo_procedimento",
        "mensagem_horario_relativo_12h",
    ]
    # coerce e imputar
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[num_cols] = classificadores.imputador_numerico.transform(df[num_cols])

    # categóricas: preenche "MISSING" para nulos
    for c in classificadores.atributos_categoricos:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("MISSING")

    # garante todas as colunas na ordem do treino
    for col in classificadores.atributos_colunas:
        if col not in df.columns:
            # se não existe (ex.: treino esperava uma coluna, mas aqui não veio), crie com 0/“MISSING”
            if col in classificadores.atributos_categoricos:
                df[col] = "MISSING"
            else:
                df[col] = 0.0
    df = df[classificadores.atributos_colunas]
    return df


@lru_cache(maxsize=128)
def obter_template_embedding_por_nome(template_nome: str) -> np.ndarray:
    resultado_query = (
        make_bq_client()
        .query(f"""
        SELECT embedding
        FROM `ip_mensageria_camada_prata.templates_embeddings`
        WHERE template_nome = '{template_nome}';
    """)
        .result()
    )
    if (
        not isinstance(resultado_query, _EmptyRowIterator)
        and resultado_query.total_rows > 0
    ):
        return np.array(next(resultado_query).embedding, dtype=float)
    raise HTTPException(
        status_code=HTTPStatus.NOT_FOUND,
        detail="Not Found :: Template não encontrado. Tente enviar o texto completo por meio do atributo `template` do objeto `Mensagem`.",
    )


@lru_cache(maxsize=128)
def obter_template_embedding_por_texto(
    template_texto: str,
    botao0_texto: Optional[str] = None,
    botao1_texto: Optional[str] = None,
    botao2_texto: Optional[str] = None,
) -> np.ndarray:
    """
    TODO: Substituir por chamada ao seu serviço de embeddings.
    Enquanto isso, devolve zeros (mesmo d do treino).
    """
    texto = "\n".join(
        [
            template_texto,
            botao0_texto or "",
            botao1_texto or "",
            botao2_texto or "",
        ]
    )
    resultado_query = make_bq_client().query(f"""
        SELECT embedding
        FROM `ip_mensageria_camada_prata.templates_embeddings`
        WHERE content = '{texto}'
    """)
    if (
        isinstance(resultado_query, _EmptyRowIterator)
        or resultado_query.total_rows == 0
    ):
        resultado_query = (
            make_bq_client()
            .query(f"""
            SELECT embedding
            FROM AI.GENERATE_EMBEDDING(
                MODEL `modelos.multimodalembedding`,
                (SELECT '{texto}' as content),
                STRUCT(128 AS output_dimensionality)
            );
        """)
            .result()
        )
        if (
            isinstance(resultado_query, _EmptyRowIterator)
            or resultado_query.total_rows == 0
        ):
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=(
                    "Internal Server Error :: Não foi possível obter o embedding.",
                ),
            )
    return np.array(next(resultado_query).embedding, dtype=float)


@lru_cache(maxsize=128)
def obter_midia_embedding(url: Optional[AnyUrl]) -> np.ndarray:
    if str(url).startswith("gs://"):
        resultado_query = (
            make_bq_client()
            .query(f"""
            SELECT embedding
            FROM `ip_mensageria_camada_prata.templates_midias_embeddings`
            WHERE ref.uri = '{url}'
        """)
            .result()
        )
    elif str(url).startswith("http"):
        resultado_query = (
            make_bq_client()
            .query(f"""
            SELECT embedding
            FROM `{BQ_PROJETO}.ip_mensageria_camada_prata.templates_midias_embeddings` e
            INNER JOIN `{BQ_PROJETO}.ip_mensageria_camada_bronze.templates_midias` t
                ON REGEXP_REPLACE(t.gcs_referencia.uri, r'ip-mensageria-turn-midias/ip-mensageria-turn-midias/o/', 'ip-mensageria-turn-midias/') = e.ref.uri
            WHERE t.url_turn = '{url}'
        """)
            .result()
        )
    else:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Bad Request :: O schema da URL não é válido.",
        )
    if (
        isinstance(resultado_query, _EmptyRowIterator)
        or resultado_query.total_rows == 0
    ):
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=(
                "Not Found :: Não foi possível obter o "
                "embedding da mídia com a URL fornecida.",
            ),
        )
    return np.array(next(resultado_query).embedding, dtype=float)


def converter_df_em_pool(df: pd.DataFrame, classificadores: Classificador) -> Pool:
    cat_idx = [
        classificadores.atributos_colunas.index(c)
        for c in classificadores.atributos_categoricos
        if c in classificadores.atributos_colunas
    ]
    return Pool(df, cat_features=cat_idx)


def thompson_sample(p: float, se: float) -> float:
    a, b = beta_from_mean_se(p, se)
    return float(np.random.beta(a, b))
