from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.database.session import get_db
from backend.app.models.all_models import User
from backend.app.schemas.all_schemas import UserRegister, UserLogin, TokenResponse, UserResponse
from backend.app.core.security import get_password_hash, verify_password, create_access_token, get_current_user_optional

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(user_in: UserRegister, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.username == user_in.username))
    if res.scalars().first():
        raise HTTPException(status_code=400, detail="Username already exists.")

    user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password)
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


@router.post("/token", response_model=TokenResponse)
@router.post("/login", response_model=TokenResponse)
async def login(user_in: UserLogin, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.username == user_in.username))
    user = res.scalars().first()

    if not user or not verify_password(user_in.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user_optional)):
    if not current_user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return UserResponse.model_validate(current_user)
