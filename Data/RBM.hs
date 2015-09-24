{-# LANGUAGE BangPatterns #-}
module Data.RBM(rbm
               ,RBM
               ,contrastiveDivergence
               ,energy
               ,hiddenProbs
               ,inputProbs
               ,sample
               ,perf
               ,test
               ) where

--benchmark modules
import Criterion.Main(defaultMainWith,defaultConfig,bgroup,bench,whnf)
import Criterion.Types(reportFile,timeLimit)
--test modules
import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess,maxSize)
import Data.Word(Word8)
--impl modules
--
import qualified System.Random as R

import Control.Monad.Identity(runIdentity)
import Control.Monad(foldM)
import Debug.Trace(trace)
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

-- |create an rbm with some randomized weights
rbm :: RandomGen r => r -> Int -> Int -> RBM
rbm r ni nh = M.randomish (ni, nh) (-0.01, 0.01) (fst $ random r)

-- | given an rbm and a biased input array, generate the energy
energy :: (Monad m) => RBM -> (Matrix U B I) -> m Double
energy rb bxi = do
   bxh <- hiddenProbs rb bxi
   ixb <- M.transpose bxi
   ixh <- ixb `M.mmult` bxh
   enr <- (M.sum $ rb *^ ixh)
   return $ negate enr

-- |given an unbiased input batch, generate the the RBM weight updates
contrastiveDivergence:: (Monad m) => Int -> Matrix U I H -> Matrix U B I -> m (Matrix U I H, Double)
contrastiveDivergence seed ixh bxi = do
   !wd <- weightDiff seed ixh bxi
   !uave <- M.sum $ M.map abs wd
   !wave <- M.sum $ M.map abs ixh
   let lc = rate par
       lc' = if wave > uave || uave == 0 
               then lc 
               else (wave / uave) * lc 
   let wd' = M.map ((*) lc') wd
   urbm <- M.d2u $ rbm' +^ wd'
   err <- M.mse wd'
   return (urbm, err)
{-# INLINE weightUpdate #-}

weightDiff :: Monad m => Int -> Matrix U I H -> Matrix U B I -> m (Matrix U I H)
weightDiff seed ixh bxi = do
   let (r1:r2:_) = randoms seed
   bxh <- sample r1 <=< hiddenProbs ixh bxi
   ixb' <- sample r1 <=< inputProbs ixh bxh
   ixb <- M.transpose bxi
   w1 <- ixb `M.mmult` bxh
   w2 <- ixb' `M.mmult` bxh
   M.d2u $ w1 -^ w2
{-# INLINE weightDiff #-}

{-|
 - given a biased input generate probabilities of the hidden layer
 - incuding the biased probability
 -
 - map sigmoid $ biased `mmult` weights
 -
 --}
hiddenProbs :: (Monad m) => RBM -> (Matrix U B I) -> m (Matrix U B H)
hiddenProbs ixh bxi = do
   !bxh <- bxi `M.mmult` ixh 
   let update v r _ | r == 0 = 1 -- ^ set bias output to 1
                    | otherwise = sigmoid v
   M.d2u $ M.traverse sigmoid bxh
{-# INLINE hiddenProbs #-}


{-|
 - given a batch biased hidden sample generate probabilities of the input layer
 - incuding the biased probability
 -
 - transpose of the hiddenProbs function
 -
 - map sigmoid $ (transpose inputs) `mmult` weights
 -
 --}
inputProbs :: (Monad m) => (Matrix U I H) -> (Matrix U B H) -> m (Matrix U I B)
inputProbs ixh bxh = do
   !ixb <- ixh `M.mmultT` bxh
   let update v _ c | c == 0 = 1 -- ^ set bias output to 1
                    | otherwise = sigmoid v
   d2u $ M.traverse sigmoid ixb
{-# INLINE inputProbs #-}

sample :: (Monad m) => Int -> Matrix U a b -> m (Matrix U a b) 
sample seed axb = do
   let rands = M.randomish (M.shape axb) (0,1) seed 
   M.d2u $ M.zipWith checkP ixb rands
{-# INLINE sample #-}

randoms :: Int -> [Int] 
randoms seed = R.randoms (R.mkStdGen seed)
{-# INLINE randoms #-}

--sample is 0 if generated number gg is greater then probabiliy pp
--so the higher pp the more likely that it will generate a 1
checkP ::  Double -> Double -> Double
checkP gen rand
   | gen > rand = 1
   | otherwise = 0
{-# INLINE checkP #-}

-- sigmoid function
sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))
{-# INLINE sigmoid #-}

-- test to see if we can learn a random string
run_prop_learned :: Double -> Int -> Int -> Double
run_prop_learned lrate ni nh = runIdentity $ do
   let rb = rbm r1 (fi ni) (fi nh)
       inputbatchL = concat $ replicate batchsz inputlst
       inputbatch = M.fromList (batchsz,fi ni) $ inputbatchL
       geninputs = randomRs (0::Int,1::Int) r2
       inputlst = map fromIntegral $ take (fi ni) $ 1:geninputs
       fi ww = 1 + ww
       (r1:r2,r3:_) = randoms (ni + nh)
       batchsz = 2000
       par = params { rate = 0.1 * lrate, miniBatch = 5, maxBatches = 2000  }
   lrb <- learn par rb [return inputbatch]
   reconErr r3 lrb [return inputbatch]

prop_learned :: Word8 -> Word8 -> Bool
prop_learned ni nh = 0.1 > (run_prop_learned 1.0 (fi ni) (fi nh))
   where
      fi = fromIntegral

prop_not_learned :: Word8 -> Word8 -> Bool
prop_not_learned ni nh = 0.1 < (run_prop_learned (-1.0) (fi ni) (fi nh))
   where
      fi ii = 2 + (fromIntegral ii)

prop_learn :: Word8 -> Word8 -> Bool
prop_learn ni nh = runIdentity $ do
   let inputs = M.fromList (fi nh, fi ni) $ take ((fi ni) * (fi nh)) $ cycle [0,1]
       seed = ni + nh
       rb = rbm seed (fi ni) (fi nh)
       fi ww = 1 + (fromIntegral ww)
       par = params { rate = 1.0 , miniBatch = 5, maxBatches = 2000 }
   lrb <- learn par rb [return $ inputs]
   return $ (M.elems rb) == (M.elems lrb)

prop_batch :: Word8 -> Word8 -> Word8 -> Bool
prop_batch ix ni nh = runIdentity $ do
   let rb = rbm seed (fi ni) (fi nh)
       seed = ix + ni + nh
       inputs = M.fromList (fi ix,fi ni) $ take ((fi ni) * (fi ix)) $ cycle [0,1]
       fi ww = 1 + (fromIntegral ww)
       par = params { rate = 1.0 , miniBatch = 5, maxBatches = 2000  }
   lrb <- batch par rb [return inputs]
   return $ (M.elems rb) == (M.elems lrb)

prop_init :: Word8 -> Word8 -> Bool
prop_init ni nh = (fi ni) * (fi nh)  == (M.elems rb)
   where
      seed = ni + nh
      rb = rbm seed (fi ni) (fi nh)
      fi :: Word8 -> Int
      fi ww = 1 + (fromIntegral ww)

prop_hiddenProbs :: Int -> Word8 -> Word8 -> Bool
prop_hiddenProbs seed ni nh = runIdentity $ do
   let rb = rbm seed (fi ni) (fi nh)
       fi ww = 1 + (fromIntegral ww)
       input = M.randomish (1, (fi ni)) (0,1) seed
   pp <- hiddenProbs rb input
   return $ (fi nh) == (M.col pp)

prop_hiddenProbs2 :: Bool
prop_hiddenProbs2 = runIdentity $ do
   let h0 = w00 * i0 + w10 * i1
       h1 = w01 * i0 + w11 * i1
       h2 = w02 * i0 + w12 * i1
       i0:i1:_ = [1..]
       w00:w01:w02:w10:w11:w12:_ = [1..]
       wws = [w00,w01,w02,w10,w11,w12]
       input = M.fromList (1,2) $ [i0,i1]
       rb = M.fromList (2,3) wws
   pp <- M.toList <$> hiddenProbs rb input
   return $ pp == map sigmoid [h0, h1, h2]

prop_inputProbs :: Int -> Word8 -> Word8 -> Bool
prop_inputProbs seed ni nh = runIdentity $ do
   let hidden = M.randomish (1,(fi nh)) (0,1) seed
       rb = rbm seed (fi ni) (fi nh)
       fi ww = 1 + (fromIntegral ww)
   pp <- inputProbs rb hidden
   return $ (fi ni) == (M.row pp)

prop_inputProbs2 :: Bool
prop_inputProbs2 = runIdentity $ do
   let i0 = w00 * h0 + w10 * h1
       i1 = w01 * h0 + w11 * h1
       i2 = w02 * h0 + w12 * h1
       h0:h1:_ = [1..]
       w00:w01:w02:w10:w11:w12:_ = [1..]
       wws = [w00,w01,w02,w10,w11,w12]
       hiddens = M.fromList (1,2) [h0,h1]
       rb = M.fromList (2,3) $ wws
   rb' <- M.transpose rb
   pp <- inputProbs rb' hiddens
   pp' <- M.toList <$> M.transpose pp
   return $ pp' == map sigmoid [i0,i1,i2]

prop_energy :: Int -> Word8 -> Word8 -> Bool
prop_energy seed ni nh = runIdentity $ do
   let input = M.randomish (1, (fi ni)) (0,1) seed
       rb = rbm seed (fi ni) (fi nh)
       fi ww = 1 + (fromIntegral ww)
   ee <- energy rb input
   return $ not $ isNaN ee

test :: IO ()
test = do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 100, maxSize = 10 }
       runtest tst p =  do putStrLn tst; check =<< verboseCheckWithResult cfg p
   runtest "init"         prop_init
   runtest "energy"       prop_energy
   runtest "hiddenp"      prop_hiddenProbs
   runtest "hiddenp2"     prop_hiddenProbs2
   runtest "inputp"       prop_inputProbs
   runtest "inputp2"      prop_inputProbs2
   runtest "batch"        prop_batch
   runtest "learn"        prop_learn
   runtest "notlearnred"  prop_not_learned
   runtest "learned"      prop_learned

perf :: IO ()
perf = do
   let file = "dist/perf-repa-RBM.html"
       cfg = defaultConfig { reportFile = Just file, timeLimit = 1.0 }
   defaultMainWith cfg [
       bgroup "energy" [ bench "63x63"  $ whnf (prop_energy 0 63) 63
                       , bench "127x127"  $ whnf (prop_energy 0 127) 127
                       , bench "255x255"  $ whnf (prop_energy 0 255) 255
                       ]
      ,bgroup "hidden" [ bench "63x63"  $ whnf (prop_hiddenProbs 0 63) 63
                       , bench "127x127"  $ whnf (prop_hiddenProbs 0 127) 127
                       , bench "255x255"  $ whnf (prop_hiddenProbs 0 255) 255
                       ]
      ,bgroup "input" [ bench "63x63"  $ whnf (prop_inputProbs 0 63) 63
                      , bench "127x127"  $ whnf (prop_inputProbs 0 127) 127
                      , bench "255x255"  $ whnf (prop_inputProbs 0 255) 255
                      ]
      ,bgroup "batch" [ bench "15"  $ whnf (prop_batch 15 15) 15
                      , bench "63x63"  $ whnf (prop_batch 63 63) 63
                      , bench "127x127"  $ whnf (prop_batch 127 127) 127
                      , bench "255x255"  $ whnf (prop_batch 255 255) 255
                      ]
      ]
   putStrLn $ "perf log written to " ++ file
