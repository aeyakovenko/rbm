{-# LANGUAGE BangPatterns #-}
module Data.RBM.State where

import qualified Data.RBM as R
import qualified Control.Monad.Trans.State.Strict as S
import qualified Data.Matrix as M
import Control.Monad.Loops(whileM_)
import Data.Matrix(Matrix(..)
                  ,(-^)
                  ,U
                  ,I
                  ,B
                  )


data RBMS = RBMS { _rbm :: R.RBM
                 , _seed :: Int
                 , _count :: Int
                 }

run :: Monad m => R.RBM -> Int -> S.StateT RBMS m a -> m (a, R.RBM)
run rbm seed action = do
   (a,rbms) <- S.runStateT action (RBMS rbm seed 0)
   return (a, _rbm rbms)

train :: Monad m => Double -> Int -> (Double -> Bool) -> Matrix U B I -> S.StateT RBMS m ()
train lc n f ins = do
   let notdone = (&&) <$> (not <$> f <$> reconErr ins) <*> ((n>) <$> count)
   whileM_ notdone (contraDiv lc ins)

-- |Run Constrastive Divergance learning in the State monad
contraDiv :: Monad m => Double -> Matrix U B I -> S.StateT RBMS m ()
contraDiv lc bxi = do 
   (RBMS !ixh seed cnt) <- S.get 
   !uixh <- R.contraDiv lc ixh seed bxi
   S.put (RBMS uixh (seed + 1) (cnt + 1))

-- |Compute the input reconstruction error with the current RBM in the state.
reconErr :: Monad m => Matrix U B I -> S.StateT RBMS m Double
reconErr bxi = do
   rbms <- S.get 
   bxi' <- R.reconstruct (_rbm rbms) bxi
   M.mse $ bxi' -^ bxi

-- |Return how many times we have executed contraDiv
count :: Monad m => S.StateT RBMS m Int
count = do
   rbms <- S.get 
   return $ _count rbms
 
