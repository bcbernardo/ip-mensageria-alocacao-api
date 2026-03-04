from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import wraps
from http import HTTPStatus
from typing import Callable, ParamSpec, TypeVar

from fastapi import Header, HTTPException
from jose import JWTError, jwt
from passlib.context import CryptContext

from ip_mensageria_alocacao_api.core import configs
from ip_mensageria_alocacao_api.core.bd import make_bq_client
from ip_mensageria_alocacao_api.core.modelos import TokenDados, UsuarioNaBase

P = ParamSpec("P")
R = TypeVar("R")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verificar_senha(plain_password: str, senha_hash: str) -> bool:
    return pwd_context.verify(plain_password, senha_hash)


def obter_hash_senha(password: str) -> str:
    return pwd_context.hash(password)


def obter_usuario(usuario_nome: str | None) -> UsuarioNaBase | None:
    if not usuario_nome:
        return None
    query = f"""
        SELECT usuario, senha_hash, desativado
        FROM `ip_mensageria_camada_ouro.usuarios_api_predicao`
        WHERE usuario = '{usuario_nome}'
    """
    resultado_query = make_bq_client().query(query).result()
    if resultado_query.total_rows == 0:
        return None
    usuario_linha = next(resultado_query)
    return UsuarioNaBase(
        usuario_nome=usuario_linha.usuario,
        senha_hash=usuario_linha.senha_hash,
        desativado=usuario_linha.desativado,
    )


def autenticar_usuario(usuario_nome: str, password: str) -> bool | UsuarioNaBase:
    user = obter_usuario(usuario_nome)
    if not user:
        return False
    if not verificar_senha(password, user.senha_hash):
        return False
    return user


def criar_token_acesso(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(tz=UTC) + expires_delta
    else:
        expire = datetime.now(tz=UTC) + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        configs.API_CHAVE,
        algorithm=configs.JWT_ALGORITMO,
    )
    return encoded_jwt


def obter_usuario_atual_via_api_key(
    x_api_key: str = Header(None, alias="X-Api-Key"),
) -> UsuarioNaBase:
    if not x_api_key:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Bad Request :: No api key provided",
        )
    credentials_exception = HTTPException(
        status_code=HTTPStatus.UNAUTHORIZED,
        detail="Unauthorized access: bad X-Api-Key",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            x_api_key,
            configs.API_CHAVE,
            algorithms=[configs.JWT_ALGORITMO],
        )
        usuario_nome = payload.get("sub")

        if usuario_nome is None:
            raise credentials_exception
        token_data = TokenDados(usuario_nome=usuario_nome)

    except JWTError:
        raise credentials_exception

    user = obter_usuario(usuario_nome=token_data.usuario_nome)

    if user is None:
        raise credentials_exception
    return user


def validar_token(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def decorator(*args: P.args, **kwargs: P.kwargs) -> R:
        token = kwargs.get("X-Api-Key")
        if not isinstance(token, str) or not token:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Bad Request :: No api key provided",
            )
        obter_usuario_atual_via_api_key(token)
        return func(*args, **kwargs)

    return decorator
