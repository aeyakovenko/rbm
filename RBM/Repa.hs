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
                    ,mkStdGen
                    ,split
                    )

import Control.Monad.Identity(runIdentity)

data RBM = RBM { weights :: Array U DIM2 Double -- weight matrix with 1 bias nodes in each layer, numHidden + 1 x numInputs  + 1
               }

numHidden :: RBM -> Int
numHidden rb = (row $ R.extent $ weights rb) 

numInputs :: RBM -> Int
numInputs rb = (col $ R.extent $ weights rb) 

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
rbm r ni nh = RBM nw
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
learn rand rb iis = learn r2 nrb (tail iis)
   where
      (r1,r2) = split rand
      nrb = batch r1 rb (head iis)
{-# INLINE learn #-}

-- given a batch of unbiased inputs, update the rbm weights from the batch at once 
batch :: RandomGen r => r -> RBM -> Array U DIM2 Double -> RBM
batch rand rb inputs = uw `R.deepSeqArray` rb { weights = uw }
   where
      uw = R.computeUnboxedS $ R.zipWith (+) (weights rb) wd
      sh = R.extent $ weights rb
      wd = weightUpdatesLoop rand rb inputs 0 (R.fromListUnboxed sh [0..])
{-# INLINE batch #-}

-- given a batch of unbiased inputs, generate the the RBM weight updates for the batch
weightUpdatesLoop :: RandomGen r => r -> RBM -> Array U DIM2 Double -> Int -> Array U DIM2 Double -> Array U DIM2 Double
weightUpdatesLoop _rand _rb inputs rx wds
   | (row $ R.extent inputs) == rx = wds
weightUpdatesLoop rand rb inputs rx pwds = weightUpdatesLoop r2 rb inputs (rx + 1) nwds
   where
      biased = R.computeUnboxedS $ R.slice inputs (Any :. rx :. All)  
      nwds = pwds `R.deepSeqArray` (R.computeUnboxedS $ R.zipWith (+) pwds wd)
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

-- 
-- -- tests
-- 
-- -- test to see if we can learn a random string
-- prop_learned :: Word8 -> Word8 -> Bool
-- prop_learned ni' nh' = (tail regened) == input
--    where
--       regened = regenerate (mr 2) lrb $ generate (mr 3) lrb (1:input)
--       --learn the inputs
--       lrb = learn (mr 1) rb [inputs]
--       rb = rbm (mr 0) ni nh
--       inputs = replicate 2000 $ input
--       --convert a random list of its 0 to 1 to doubles
--       input = map fromIntegral $ take ni $ randomRs (0::Int,1::Int) (mr 4)
--       ni = fromIntegral ni' :: Int
--       nh = fromIntegral nh' :: Int
--       --creates a random number generator with a seed
--       mr i = mkStdGen (ni + nh + i)
-- 
-- 
-- prop_learn :: Word8 -> Word8 -> Bool
-- prop_learn ni nh = ln == (length $ weights $ lrb)
--    where
--       ln = ((fi ni) + 1) * ((fi nh) + 1)
--       lrb = learn' rb 1
--       learn' rr ix = learn (mkStdGen ix) rr [[(take (fi ni) $ cycle [0,1])]]
--       rb = rbm (mkStdGen 0) (fi ni) (fi nh)
--       fi = fromIntegral
-- 
-- prop_batch :: Word8 -> Word8 -> Word8 -> Bool
-- prop_batch ix ni nh = ln == (length $ weights $ lrb)
--    where
--       ln = ((fi ni) + 1) * ((fi nh) + 1)
--       lrb = batch rand rb inputs
--       rb = rbm rand (fi ni) (fi nh)
--       rand = mkStdGen ln
--       inputs = replicate (fi ix) $ take (fi ni) $ cycle [0,1]
--       fi = fromIntegral

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

-- prop_hiddenProbs2 :: Bool
-- prop_hiddenProbs2 = pp == map sigmoid [h0, h1]
--    where
--       h0 = w00 * i0 + w01 * i1 + w02 * i2  
--       h1 = w10 * i0 + w11 * i1 + w12 * i2 
--       i0:i1:i2:_ = [1..]
--       w00:w01:w02:w10:w11:w12:_ = [1..]
--       wws = [w00,w01,w02,w10,w11,w12]
--       pp = hiddenProbs rb [i0,i1,i2]
--       rb = RBM wws 2 1
-- 
-- prop_inputProbs :: Int -> Word8 -> Word8 -> Bool
-- prop_inputProbs gen ni nh = (fi ni) + 1 == length pp
--    where
--       pp = inputProbs rb $ replicate ((fi nh) + 1) 0.0
--       rb = rbm (mkStdGen gen) (fi ni) (fi nh)
--       fi = fromIntegral
-- 
-- prop_inputProbs2 :: Bool
-- prop_inputProbs2 = pp == map sigmoid [i0,i1,i2]
--    where
--       i0 = w00 * h0 + w10 * h1
--       i1 = w01 * h0 + w11 * h1
--       i2 = w02 * h0 + w12 * h1
--       h0:h1:_ = [1..]
--       w00:w01:w02:w10:w11:w12:_ = [1..]
--       wws = [w00,w01,w02,w10,w11,w12]
--       pp = inputProbs rb [h0,h1]
--       rb = RBM wws 2 1
-- 
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
   --runtest "hiddenp2" prop_hiddenProbs2
   --runtest "inputp"   prop_inputProbs
   --runtest "inputp2"  prop_inputProbs2
   --runtest "learn"    prop_learn
   --runtest "batch"    prop_batch
   --runtest "learned"  prop_learned

perf :: IO ()
perf = do
   let file = "dist/perf-repa-RBM.html"
       cfg = defaultConfig { reportFile = Just file, timeLimit = 1.0 }
   defaultMainWith cfg [
       bgroup "energy" [ bench "64x64"  $ whnf (prop_energy 0 64) 64
                       , bench "128x128"  $ whnf (prop_energy 0 128) 128
                       , bench "255x255"  $ whnf (prop_energy 0 255) 255
                       ]
   --   ,bgroup "hidden" [ bench "64x64"  $ whnf (prop_hiddenProbs 0 64) 64
   --                    , bench "128x128"  $ whnf (prop_hiddenProbs 0 128) 128
   --                    , bench "255x255"  $ whnf (prop_hiddenProbs 0 255) 255
   --                    ]
   --   ,bgroup "input" [ bench "64x64"  $ whnf (prop_inputProbs 0 64) 64
   --                   , bench "128x128"  $ whnf (prop_inputProbs 0 128) 128
   --                   , bench "255x255"  $ whnf (prop_inputProbs 0 255) 255
   --                   ]
   --   ,bgroup "batch" [ bench "64x64"  $ whnf (prop_batch 64 64) 64
   --                   , bench "128x128"  $ whnf (prop_batch 128 128) 128
   --                   , bench "255x255"  $ whnf (prop_batch 255 255) 255
   --                   ]
      ]
   putStrLn $ "perf log written to " ++ file
