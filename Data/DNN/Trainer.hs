{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
module Data.DNN.Trainer where

import qualified Data.RBM as R
import qualified Data.NN as N
import qualified Control.Monad.State.Strict as S
import qualified Control.Monad.Except as E
import qualified Data.Matrix as M
import Control.Monad(when) 
import Data.Matrix(Matrix(..)
                  ,(-^)
                  ,U
                  ,I
                  ,H
                  ,B
                  )

data DNNS = DNNS { _nn :: [Matrix U I H]
                 , _layer :: Int
                 , _seed :: Int
                 , _count :: Int
                 , _lr :: Double
                 }

type Trainer m a = E.ExceptT a (S.StateT DNNS m) a

finish :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
       => a -> m a
finish v = E.throwError v

finish_ :: (Monad m, E.MonadError () m, S.MonadState DNNS m) 
       => m ()
finish_ = finish ()

run :: Monad m => Trainer m a -> m (a, R.RBM)
run action = do
   (a,dnns) <- S.runStateT (E.runExceptT action) (DNNS [] 0 0 0.001)
   let unEither (Left v) = v
       unEither (Right v) = v
   return (unEither a, _nn dnns)

-- |Run feedForward MLP algorithm over the entire DNN.
feedForward :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
            => Matrix U B I -> m (Matrix U B H)
feedForward !bxi = do
   nn <- getDNN
   N.feedForward nn bxi

-- |Run Back Propagation training over the entire DNN.
backProp :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
         => Matrix U B I -> Matrix U B H -> m Double
backProp !bxi !bxh= do
   dnns <- S.get
   lr <- getLearnRate
   _ <- incCount
   !(unn,err) <- N.backPropagate (_nn dnns) lr bxi bxh
   S.put dnns { _nn  = unn }
   return err

-- |Run Constrastive Divergance on the last layer in the DNN
contraDiv :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
          => Matrix U B I -> m ()
contraDiv !bxi = do 
   seed <- nextSeed
   cnt <- incCount
   lr <- getLearnRate
   ixh <- popLastLayer
   !uixh <- R.contraDiv lr ixh seed bxi
   pushLastLayer uixh

-- |Add the first layer to the DNN.
initFirstLayer :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
               => Int -> Int -> m ()
initFirstLayer ni nh = do
   s <- nextSeed
   pushLastLayer $ R.new s ni nh

-- |Add a layer as the new output layer.
addLayer :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
         => Int -> m ()
addLayer num = do
   s <- nextSeed
   dnns <- S.get
   ixh <- getLastLayer
   S.put $ dnns { _nn = (_nn dnns) ++ [R.new s (R.col ixh) num] }

-- |Compute the input reconstruction error with the current RBM in the state.
reconErr :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
         => Matrix U B I -> m Double
reconErr bxi = do
   bxi' <- reconstruct bxi
   M.mse $ bxi' -^ bxi

-- |Reconstruct the input with the current RBM
reconstruct :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
            => Matrix U B I -> m (Matrix U B I)
reconstruct bxi = do
   dnns <- S.get
   R.reconstruct bxi (_nn dnns)

-- |Resample the input with the current RBM
resample :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
         => Matrix U B I -> m (Matrix U B I)
resample bxi = do
   dnns <- S.get
   R.resample bxi (_nn dnns)

-- |Return how many times we have executed contraDiv or backProp
getCount :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
      => m Int
getCount = do
   dnns <- S.get 
   return $ _count dnns

-- |Set the count to a specific value.
setCount :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
      => Int -> m ()
setCount n = do
   S.get >>= \ x -> S.put x { _count = n }

-- |Increment the count and the return the previous value.
incCount :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
        => m Int
incCount = do
   v <- S.get
   S.put v { _count = (_count v) + 1 }
   return $ _count v

-- |Return the DNN
getDNN :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
        => m ([Matrix U I H])
getDNN = _nn <$> S.get

-- |Pop the last layer of the DNN.
popLastLayer :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
        => m (Matrix U I H)
popLastLayer = do
   v <- S.get
   let check [] = error "Data.DNN.Trainer.popLastLayer: empty dnn, call initFirstLayer first."
       check ls = ls
       (rest,l) = splitAt ((length ls) -1) $ check (_nn v)
   S.put v { _nn = rest }
   return $ head $ check l

-- |Push the updated layer as the last in the DNN.
pushLastLayer :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
                => (Matrix U I H) -> m ()
pushLastLayer ixh = do
   v <- S.get
   S.put v { _nn = reverse $ ixh:(tail $ reverse (_nn v)) }

-- |Get the next random seed.
nextSeed :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
         => m Int
nextSeed = do
   S.get >>= \ x -> S.put x { _seed = (_seed x) + 1 }
   _seed <$> S.get

-- |Set the learning rate used in backProp and contraDiv.
setLearnRate :: (Monad m, E.MonadError a m, S.MonadState DNNS m) 
             => Double -> m ()
setLearnRate d = S.get >>= \ x -> S.put x { _lr = d }

-- |Get the current learning rate.
getLearnRate :: (Monad m, E.MonadError a m, S.MonadState RBMS m) 
             => m Double
getLearnRate = _lr <$> S.get 
