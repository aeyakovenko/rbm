{-# LANGUAGE BangPatterns #-}
module Data.RBM(newRBM
               ,RBM
               ,contraDiv
               ,contraDivS
               ,energy
               ,hiddenPs
               ,inputPs
               ,sample
               ,reconstruct
               ) where

import qualified System.Random as R
import qualified Control.Monad.Trans.State.Strict as S
import qualified Data.Matrix as M

import Data.Matrix(Matrix(..)
                  ,(*^)
                  ,(+^)
                  ,(-^)
                  ,U
                  ,I
                  ,H
                  ,B
                  )

-- | weight matrix, numInputs x numHidden
-- | where the bias node is the first node
type RBM = Matrix U I H

-- |Create an rbm with some randomized weights
newRBM :: Int -> Int -> Int -> RBM
newRBM seed ni nh = M.randomish (ni, nh) (-0.01, 0.01) seed

-- |Compute the energy of the RBM with the batch of input.
energy :: (Monad m) => RBM -> (Matrix U B I) -> m Double
energy rb bxi = do
   bxh <- hiddenPs rb bxi
   ixb <- M.transpose bxi
   ixh <- ixb `M.mmult` bxh
   enr <- (M.sum $ rb *^ ixh)
   return $ negate enr

-- |Reconstruct the input
reconstruct :: Monad m => Matrix U I H -> Matrix U B I -> m (Matrix U B I)
reconstruct ixh ins = M.transpose =<< inputPs ixh =<< hiddenPs ixh ins
 
-- |Run Constrastive Divergance learning in the State monad
contraDivS :: Monad m => Double -> Matrix U B I -> S.StateT (Matrix U I H, Int) m Double
contraDivS lc bxi = do 
   (!ixh,seed) <- S.get 
   (!uixh, err) <- contraDiv lc ixh seed bxi
   S.put (uixh, seed + 1)
   return err

-- |Run Contrastive Divergance learning.  Return the updated RBM
contraDiv :: (Monad m) => Double -> (Matrix U I H) -> Int -> Matrix U B I -> m (Matrix U I H, Double)
contraDiv lc ixh seed bxi = do
   !wd <- weightDiff seed ixh bxi
   !uave <- M.sum $ M.map abs wd
   !wave <- M.sum $ M.map abs ixh
   let lc' = if wave > uave || uave == 0 
               then lc 
               else (wave / uave) * lc 
   let wd' = M.map ((*) lc') wd
   urbm <- M.d2u $ ixh +^ wd'
   err <- M.mse wd'
   return (urbm, err)
{-# INLINE contraDiv #-}

weightDiff :: Monad m => Int -> Matrix U I H -> Matrix U B I -> m (Matrix U I H)
weightDiff seed ixh bxi = do
   let (s1:s2:_) = seeds seed
   bxh <- sample s1 =<< hiddenPs ixh bxi
   ixb' <- sample s2 =<< inputPs ixh bxh
   ixb <- M.transpose bxi
   w1 <- ixb `M.mmult` bxh
   w2 <- ixb' `M.mmult` bxh
   M.d2u $ w1 -^ w2
{-# INLINE weightDiff #-}

{-|
 - Given a biased input generate probabilities of the hidden layer
 - incuding the biased probability.
 --}
hiddenPs :: (Monad m) => RBM -> (Matrix U B I) -> m (Matrix U B H)
hiddenPs ixh bxi = do
   !bxh <- bxi `M.mmult` ixh 
   let update v r _ | r == 0 = 1 -- ^ set bias output to 1
                    | otherwise = sigmoid v
   M.d2u $ M.traverse update bxh
{-# INLINE hiddenPs #-}


{-|
 - Given a batch biased hidden sample generate probabilities of the input layer
 - incuding the biased probability.
 --}
inputPs :: (Monad m) => (Matrix U I H) -> (Matrix U B H) -> m (Matrix U I B)
inputPs ixh bxh = do
   !ixb <- ixh `M.mmultT` bxh
   let update v _ c | c == 0 = 1 -- ^ set bias output to 1
                    | otherwise = sigmoid v
   M.d2u $ M.traverse update ixb
{-# INLINE inputPs #-}

-- |sample the matrix of probabilities
sample :: (Monad m) => Int -> Matrix U a b -> m (Matrix U a b) 
sample seed axb = do
   let rands = M.randomish (M.shape axb) (0,1) seed 
   M.d2u $ M.zipWith checkP axb rands
{-# INLINE sample #-}

seeds :: Int -> [Int] 
seeds seed = R.randoms (R.mkStdGen seed)
{-# INLINE seeds #-}

-- | Generate a sample for each value.  If the random number is less
-- | then the value return 1, otherwise return 0.
checkP ::  Double -> Double -> Double
checkP gen rand
   | gen > rand = 1
   | otherwise = 0
{-# INLINE checkP #-}

-- | sigmoid function
sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))
{-# INLINE sigmoid #-}
