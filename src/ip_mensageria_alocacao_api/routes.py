import logging
from datetime import timedelta
from http import HTTPStatus
from typing import Sequence

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm

from ip_mensageria_alocacao_api.apis import (
    alocar_entre_mensagens,
    prever_probabilidade_mensagem_ser_efetiva,
)
from ip_mensageria_alocacao_api.core import configs
from ip_mensageria_alocacao_api.core.autenticacao import (
    autenticar_usuario,
    criar_token_acesso,
    obter_usuario_atual_via_api_key,
)
from ip_mensageria_alocacao_api.core.classificadores import carregar_classificadores
from ip_mensageria_alocacao_api.core.modelos import (
    LinhaCuidado,
    Mensagem,
    MensagemTipo,
    Predicao,
    PredicaoSimulacao,
    Token,
    UsuarioNaBase,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
async def index() -> dict[str, str]:
    return {
        "info": "Bem-vindo(a) à API de predições e alocação "
        "Acesse o endpoint '/docs' para ver a documentação da API.",
    }


@router.post("/token", response_model=Token)
async def login_para_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> dict[str, str]:
    user = autenticar_usuario(form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Incorrect usuario_nome or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(
        seconds=configs.TOKEN_VALIDADE_MINUTOS,
    )
    access_token = criar_token_acesso(
        data={"sub": user.usuario_nome},  # type: ignore
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/prever_efetividade_mensagem", response_model=Predicao)
async def prever_efetividade_mensagem(
    cidadao_id: str,
    linha_cuidado: LinhaCuidado,
    mensagem_tipo: MensagemTipo,
    mensagem: Mensagem,
    request: Request,
    usuario: UsuarioNaBase = Depends(obter_usuario_atual_via_api_key),
) -> Predicao:
    try:
        classificadores = request.app.state.classificadores
    except AttributeError:
        try:
            classificadores = carregar_classificadores()
        except RuntimeError as exc:
            logger.exception("Falha ao carregar classificadores")
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Falha inesperada ao carregar classificadores")
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="Classificadores indisponiveis no momento.",
            ) from exc

        request.app.state.classificadores = classificadores

    return prever_probabilidade_mensagem_ser_efetiva(
        cidadao_id=cidadao_id,
        linha_cuidado=linha_cuidado,
        mensagem_tipo=mensagem_tipo,
        mensagem=mensagem,
        classificadores=classificadores,
    )


@router.post("/alocar")
async def alocar(
    predicoes: Sequence[Predicao],
    usuario: UsuarioNaBase = Depends(obter_usuario_atual_via_api_key),
) -> PredicaoSimulacao:
    return alocar_entre_mensagens(predicoes)
