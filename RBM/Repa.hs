module RBM.Repa(rbm
               ,learn
               ,energy
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
import Data.Array.Repa(Array
                      ,U
                      ,DIM2
                      ,DIM1
                      ,Any(Any)
                      ,Z(Z)
                      ,(:.)((:.))
                      ,All(All)
                      )
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Algorithms.Randomish as R
import qualified Data.Array.Repa.Algorithms.Matrix as R
import System.Random(RandomGen
                    ,random
                    ,randomRs
                    ,mkStdGen
                    ,split
                    )

import Control.Monad.Identity(runIdentity)

type RBM = Array U DIM2 Double -- weight matrix with 1 bias nodes in each layer, numHidden + 1 x numInputs  + 1

weights :: RBM -> Array U DIM2 Double
weights = id
{-# INLINE weights #-}

numHidden :: RBM -> Int
numHidden rb = (row $ R.extent $ weights rb) 
{-# INLINE numHidden #-}

numInputs :: RBM -> Int
numInputs rb = (col $ R.extent $ weights rb) 
{-# INLINE numInputs #-}

row :: DIM2 -> Int
row (Z :. r :. _) = r
{-# INLINE row #-}

col :: DIM2 -> Int
col (Z :. _ :. c) = c
{-# INLINE col #-}

len :: DIM1 -> Int
len (Z :. i ) = i
{-# INLINE len #-}

--create an rbm with some randomized weights
rbm :: RandomGen r => r -> Int -> Int -> RBM
rbm r ni nh = nw
   where
      nw = R.randomishDoubleArray (Z :. nh :. ni) 0 1 (fst $ random r)

{-- 
 - given an rbm and a biased input array, generate the energy
 - should be: negate $ sumAll $ weights *^ (hidden `tensor` biased)
 - but everything is unrolled to experiment with Repa's parallelization
 --}
energy :: RBM -> Array U DIM1 Double -> Double
energy rb biased = negate ee
   where
      ee = runIdentity $ hhs `R.deepSeqArray` R.sumAllP $ R.fromFunction sh (computeEnergyMatrix wws biased hhs)
      wws = weights rb
      hhs = hiddenProbs rb biased
      sh = (Z :. nh :. ni)
      ni = numInputs rb
      nh = numHidden rb

computeEnergyMatrix ::  Array U DIM2 Double -> Array U DIM1 Double -> Array U DIM1 Double -> DIM2 -> Double 
computeEnergyMatrix wws biased hhs sh = ww * ii * hh
   where
      rr = row sh 
      cc = col sh 
      ww = wws `R.index` sh
      ii = biased `R.index` (Z :. cc) 
      hh = hhs `R.index` (Z :. rr)
{-# INLINE computeEnergyMatrix #-}

{--
 - given a biased input generate probabilities of the hidden layer
 - incuding the biased probability
 -
 - map sigmoid $ weights `mmult` biased
 -
 - 
 --}
hiddenProbs :: RBM -> Array U DIM1 Double -> Array U DIM1 Double
hiddenProbs rb biased = R.computeUnboxedS $ R.fromFunction shape (computeHiddenProb wws biased)
   where
      wws = weights rb
      shape = (Z :. (numHidden rb))
{-# INLINE hiddenProbs #-}

{--
 - sigmoid of the dot product of the row
 --}
computeHiddenProb :: Array U DIM2 Double -> Array U DIM1 Double -> DIM1 -> Double
computeHiddenProb wws biased xi = sigmoid sm
   where
      rw = R.slice wws (Any :. (len xi) :. All)
      sm = R.sumAllS $ R.zipWith (*) rw biased
{-# INLINE computeHiddenProb #-}

{--
 - given a biased hidden sample generate probabilities of the input layer
 - incuding the biased probability
 -
 - transpose of the hiddenProbs function
 -
 - map sigmoid $ (transpose inputs) `mmult` weights
 - 
 --} 
inputProbs :: RBM -> Array U DIM1 Double -> Array U DIM1 Double
inputProbs rb hidden = R.computeUnboxedS $ R.fromFunction shape (computeInputProb wws hidden)
   where
      shape = (Z :. (numInputs rb))
      wws = weights rb
{-# INLINE inputProbs #-}

{--
 - sigmoid of the dot product of the col
 --}
computeInputProb :: Array U DIM2 Double -> Array U DIM1 Double -> DIM1 -> Double
computeInputProb wws hidden xi = sigmoid sm
   where
      rw = R.slice wws (Any :. (len xi))
      sm = R.sumAllS $ R.zipWith (*) rw hidden
{-# INLINE computeInputProb #-}

-- update the rbm weights from each batch
learn :: RandomGen r => r -> RBM -> [Array U DIM2 Double]-> RBM
learn _ rb [] = rb
learn rand rb iis = rb `R.deepSeqArray` learn r2 nrb (tail iis)
   where
      (r1,r2) = split rand
      nrb = batch r1 rb (head iis)
{-# INLINE learn #-}

-- given a batch of unbiased inputs, update the rbm weights from the batch at once 
batch :: RandomGen r => r -> RBM -> Array U DIM2 Double -> RBM
batch rand rb inputs = uw
   where
      uw = R.computeUnboxedS $ R.zipWith (+) (weights rb) wd
      sh = R.extent $ weights rb
      wd = weightUpdatesLoop rand rb inputs 0 $ R.fromListUnboxed sh $ take (row sh * col sh) [0..]
{-# INLINE batch #-}

-- given a batch of unbiased inputs, generate the the RBM weight updates for the batch
weightUpdatesLoop :: RandomGen r => r -> RBM -> Array U DIM2 Double -> Int -> Array U DIM2 Double -> Array U DIM2 Double
weightUpdatesLoop _rand _rb inputs rx wds
   | (row $ R.extent inputs) <= rx = wds
weightUpdatesLoop rand rb inputs rx pwds = pwds `R.deepSeqArray` (weightUpdatesLoop r2 rb inputs (rx + 1) nwds)
   where
      biased = R.computeUnboxedS $ R.slice inputs (Any :. rx :. All)  
      nwds = R.computeUnboxedS $ R.zipWith (+) pwds wd
      wd = weightUpdate r1 rb biased
      (r1,r2) = split rand
{-# INLINE weightUpdatesLoop #-}

-- given an unbiased input, generate the the RBM weight updates
weightUpdate :: RandomGen r => r -> RBM -> Array U DIM1 Double -> Array U DIM2 Double
weightUpdate rand rb biased = R.computeUnboxedS $ R.zipWith (-) w1 w2
   where
      (r1,r2) = split rand
      hiddens = generate r1 rb biased
      newins = hiddens `R.deepSeqArray` regenerate r2 rb hiddens
      w1 = hiddens `tensor` biased
      w2 = hiddens `tensor` newins 
{-# INLINE weightUpdate #-}

-- given a biased input (1:input), generate a biased hidden layer sample
generate :: RandomGen r => r -> RBM -> Array U DIM1 Double -> Array U DIM1 Double
generate rand rb biased = R.computeUnboxedS $ R.fromFunction (Z:. nh) gen
   where
      gen sh = genProbs (hiddenProbs rb biased) rands sh
      rands = R.randomishDoubleArray (Z :. nh)  0 1 (fst $ random rand)
      nh = numHidden rb
{-# INLINE generate #-}

-- given a biased hidden layer sample, generate a biased input layer sample
regenerate :: RandomGen r => r -> RBM -> Array U DIM1 Double -> Array U DIM1 Double
regenerate rand rb hidden = R.computeUnboxedS $ R.fromFunction (Z :. ni) gen 
   where
      gen sh = genProbs (inputProbs rb hidden) rands sh
      rands = R.randomishDoubleArray (Z :. ni) 0 1 (fst $ random rand)
      ni = numInputs rb
{-# INLINE regenerate #-}

--sample is 0 if generated number gg is greater then probabiliy pp
--so the higher pp the more likely that it will generate a 1
genProbs :: Array U DIM1 Double -> Array U DIM1 Double -> DIM1 -> Double
genProbs probs rands sh
   | len sh == 0 = 1
   | (probs `R.index` sh) > (rands `R.index` sh) = 1
   | otherwise = 0
{-# INLINE genProbs #-}

-- row vec * col vec
-- or (r x 1) * (1 x c) -> (r x c) matrix multiply 
tensor :: Array U DIM1 Double -> Array U DIM1 Double -> Array U DIM2 Double
tensor a1 a2 = R.mmultS a1' a2'
   where
      a1' :: Array U DIM2 Double
      a1' = R.computeUnboxedS $ R.reshape (Z :. len (R.extent a1) :. 1) a1
      a2' :: Array U DIM2 Double
      a2' = R.computeUnboxedS $ R.reshape (Z :. 1 :. len (R.extent a2)) a2
{-# INLINE tensor #-}
 

-- sigmoid function
sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))
{-# INLINE sigmoid #-}


-- tests

-- test to see if we can learn a random string
prop_learned :: Word8 -> Word8 -> Bool
prop_learned ni nh = (tail $ R.toList regened) == (tail $ R.toList inputarr)
   where
      regened = regenerate (mr 2) lrb $ generate (mr 3) lrb inputarr
      --learn the inputs

      lrb = learn (mr 1) rb [inputbatch]
      --creates a random number generator with a seed

      rb = rbm (mr 0) (fi ni) (fi nh)
      inputbatch = R.fromListUnboxed (Z:. batchsz :.fi ni) $ concat $ replicate batchsz inputlst
      inputarr = R.fromListUnboxed (Z:. fi ni) $ inputlst
      inputlst = take (fi ni) $ map fromIntegral $ randomRs (0::Int,1::Int) (mr 4)
      fi ww = 1 + (fromIntegral ww)
      mr i = mkStdGen (fi ni + fi nh + i)
      batchsz = 2000


prop_learn :: Word8 -> Word8 -> Bool
prop_learn ni nh = (R.extent $ weights rb) == (R.extent $ weights $ lrb)
   where
      lrb = learn rand rb [inputs]
      inputs = R.fromListUnboxed (Z:.fi nh:.fi ni) $ take ((fi ni) * (fi nh)) $ cycle [0,1]
      rand = mkStdGen $ fi nh
      rb = rbm rand (fi ni) (fi nh)
      fi ww = 1 + (fromIntegral ww)

prop_batch :: Word8 -> Word8 -> Word8 -> Bool
prop_batch ix ni nh = (R.extent $ weights rb) == (R.extent $ weights $ lrb)
   where
      lrb = batch rand rb inputs
      rb = rbm rand (fi ni) (fi nh)
      rand = mkStdGen $ fi ix
      inputs = R.fromListUnboxed (Z:.fi ix:.fi ni) $ take ((fi ni) * (fi ix)) $ cycle [0,1]
      fi ww = 1 + (fromIntegral ww)

prop_init :: Int -> Word8 -> Word8 -> Bool
prop_init gen ni nh = (fi ni) * (fi nh)  == (length $ R.toList $ weights rb)
   where
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi :: Word8 -> Int
      fi ww = 1 + (fromIntegral ww)

prop_tensor :: Bool
prop_tensor = rv == [1*4,1*5,2*4,2*5,3*4,3*5]
   where
      rv = R.toList (a1 `tensor` a2)
      a1 :: Array U DIM1 Double
      a1 = R.fromListUnboxed (Z:.3) [1,2,3]
      a2 :: Array U DIM1 Double
      a2 = R.fromListUnboxed (Z:.2) [4,5]

prop_hiddenProbs :: Int -> Word8 -> Word8 -> Bool
prop_hiddenProbs gen ni nh = (fi nh) == (len $ R.extent pp)
   where
      pp = hiddenProbs rb input
      input = R.randomishDoubleArray (Z :. (fi ni)) 0 1 gen
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi ww = 1 + (fromIntegral ww)

prop_hiddenProbs2 :: Bool
prop_hiddenProbs2 = pp == map sigmoid [h0, h1]
   where
      h0 = w00 * i0 + w01 * i1 + w02 * i2  
      h1 = w10 * i0 + w11 * i1 + w12 * i2 
      i0:i1:i2:_ = [1..]
      w00:w01:w02:w10:w11:w12:_ = [1..]
      wws = [w00,w01,w02,w10,w11,w12]
      input = R.fromListUnboxed (Z:.3) $ [i0,i1,i2]
      pp = R.toList $ hiddenProbs rb input
      rb = R.fromListUnboxed (Z:.2:.3) $ wws

prop_inputProbs :: Int -> Word8 -> Word8 -> Bool
prop_inputProbs gen ni nh = (fi ni) == (len $ R.extent pp)
   where
      pp = inputProbs rb hidden
      hidden = R.randomishDoubleArray (Z :. (fi nh)) 0 1 gen
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi ww = 1 + (fromIntegral ww)

prop_inputProbs2 :: Bool
prop_inputProbs2 = pp == map sigmoid [i0,i1,i2]
   where
      i0 = w00 * h0 + w10 * h1
      i1 = w01 * h0 + w11 * h1
      i2 = w02 * h0 + w12 * h1
      h0:h1:_ = [1..]
      w00:w01:w02:w10:w11:w12:_ = [1..]
      wws = [w00,w01,w02,w10,w11,w12]
      hiddens = R.fromListUnboxed (Z:.2) [h0,h1]
      pp = R.toList $ inputProbs rb hiddens
      rb = R.fromListUnboxed (Z:.2:.3) $ wws

prop_energy :: Int -> Word8 -> Word8 -> Bool
prop_energy gen ni nh = not $ isNaN ee
   where
      ee = energy rb input
      input = R.randomishDoubleArray (Z :. (fi ni)) 0 1 gen
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi ww = 1 + (fromIntegral ww)

test :: IO ()
test = do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 100, maxSize = 10 }
       runtest tst p =  do putStrLn tst; check =<< verboseCheckWithResult cfg p
   runtest "init"     prop_init
   runtest "energy"   prop_energy
   runtest "tensor"   prop_tensor
   runtest "hiddenp"  prop_hiddenProbs
   runtest "hiddenp2" prop_hiddenProbs2
   runtest "inputp"   prop_inputProbs
   runtest "inputp2"  prop_inputProbs2
   runtest "batch"    prop_batch
   runtest "learn"    prop_learn
   runtest "learned"  prop_learned

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
