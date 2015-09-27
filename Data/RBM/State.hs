{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
module Data.RBM.State where

import qualified Data.RBM as R
import qualified Control.Monad.State.Strict as S
import qualified Control.Monad.Except as E
import qualified Data.Matrix as M
import Control.Monad(when) 
import Data.Matrix(Matrix(..)
                  ,(-^)
                  ,U
                  ,I
                  ,B
                  )


data RBMS = RBMS { _rbm :: R.RBM
                 , _seed :: Int
                 , _count :: Int
                 , _learnRate :: Double
                 }

type TrainT m a = E.ExceptT a (S.StateT RBMS m) a

finish :: (Monad m, E.MonadError a m, S.MonadState RBMS m) 
       => a -> m a
finish v = E.throwError v

finish_ :: (Monad m, E.MonadError () m, S.MonadState RBMS m) 
       => m ()
finish_ = finish ()

finishIf :: (Monad m, E.MonadError () m, S.MonadState RBMS m) 
      => Int -> Double -> Matrix U B I -> m ()
finishIf n e b = do 
   cnt <- count
   when (n < cnt) finish_
   err <- reconErr b
   when (e > err) finish_
   return ()

run :: Monad m => R.RBM -> TrainT m a -> m (a, R.RBM)
run rb action = do
   (a,rbms) <- S.runStateT (E.runExceptT action) (RBMS rb 0 0 0.001)
   let unEither (Left v) = v
       unEither (Right v) = v
   return (unEither a, _rbm rbms)

-- |Run Constrastive Divergance learning in the State monad
contraDiv :: (Monad m, E.MonadError a m, S.MonadState RBMS m) 
          => Matrix U B I -> m ()
contraDiv bxi = do 
   (RBMS !ixh seed cnt lc) <- S.get 
   !uixh <- R.contraDiv lc ixh seed bxi
   S.put (RBMS uixh (seed + 1) (cnt + (M.row bxi)) lc)

-- |Compute the input reconstruction error with the current RBM in the state.
reconErr :: (Monad m, E.MonadError a m, S.MonadState RBMS m) 
         => Matrix U B I -> m Double
reconErr bxi = do
   rbms <- S.get
   bxi' <- R.reconstruct bxi [(_rbm rbms)]
   M.mse $ bxi' -^ bxi

-- |Return how many times we have executed contraDiv
count :: (Monad m, E.MonadError a m, S.MonadState RBMS m) 
      => m Int
count = do
   rbms <- S.get 
   return $ _count rbms

getSeed :: (Monad m, E.MonadError a m, S.MonadState RBMS m) 
        => m Int
getSeed = do
   S.get >>= \ x -> S.put x { _seed = (_seed x) + 1 }
   _seed <$> S.get

setLearnRate :: (Monad m, E.MonadError a m, S.MonadState RBMS m) 
             => Double -> m ()
setLearnRate d = S.get >>= \ x -> S.put x { _learnRate = d }

getLearnRate :: (Monad m, E.MonadError a m, S.MonadState RBMS m) 
             => m Double
getLearnRate = _learnRate <$> S.get 
