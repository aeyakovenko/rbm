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
                      ,D
                      ,DIM2
                      ,DIM1
                      ,Any(Any)
                      ,Z(Z)
                      ,(:.)((:.))
                      ,All(All)
                      ,(*^)
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

{--
 - weight matrix with 1 bias nodes in each layer, numHidden + 1 x numInputs  + 1
 --}
type RBM = HxI U

{--
 - data types to keep track of matrix orientation
 -
 - H num hidden nodes
 - I num input nodes
 - B batch size
 --}
data HxI a = HxI { unHxI :: (Array a DIM2 Double)}
data IxH a = IxH { unIxH :: (Array a DIM2 Double)}

data IxB a = IxB { unIxB :: (Array a DIM2 Double)}
data BxI a = BxI { unBxI :: (Array a DIM2 Double)}

data HxB a = HxB { unHxB :: (Array a DIM2 Double)}
data BxH a = BxH { unBxH :: (Array a DIM2 Double)}

weights :: RBM -> HxI U
weights wws = wws
{-# INLINE weights #-}

numHidden :: RBM -> Int
numHidden rb = (row $ R.extent $ unHxI $ weights rb)
{-# INLINE numHidden #-}

numInputs :: RBM -> Int
numInputs rb = (col $ R.extent $ unHxI $ weights rb)
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
rbm r ni nh = HxI nw
   where
      nw = R.randomishDoubleArray (Z :. nh :. ni) 0 1 (fst $ random r)

{--
 - given an rbm and a biased input array, generate the energy
 - should be: negate $ sumAll $ weights *^ (hidden `tensor` biased)
 - but everything is unrolled to experiment with Repa's parallelization
 --}
energy :: RBM -> Array U DIM1 Double -> Double
energy rb inputs = negate ee
   where
      ee = R.sumAllS $ (unHxI wws) *^ ((unHxB hhs) `R.mmultS` (unBxI bxi))
      bxi = BxI $ R.reshape (Z :. 1 :. (len inputs)) inputs
      ixb = IxB $ R.reshape (Z :. (len inputs) :. 1 ) inputs
      wws = weights rb
      hhs = hiddenProbs rb ixb

{--
 - given a biased input generate probabilities of the hidden layer
 - incuding the biased probability
 -
 - map sigmoid $ biased `mmult` weights
 -
 --}
hiddenProbs :: HxI U -> IxB U -> HxB D
hiddenProbs (HxI wws) (IxB iis) = R.map sigmoid $ wws `R.mmultS` iis
{-# INLINE hiddenProbs #-}

{--
 - given a batch biased hidden sample generate probabilities of the input layer
 - incuding the biased probability
 -
 - transpose of the hiddenProbs function
 -
 - map sigmoid $ (transpose inputs) `mmult` weights
 -
 --}
inputProbs :: BxH -> HxI -> BxI
inputProbs (BxH hhs) (HxI wws) = R.computeUnboxedS $ R.map sigmoid $ hhs `R.mmultS` wws
{-# INLINE inputProbs #-}

-- update the rbm weights from each batch
learn :: RandomGen r => r -> RBM -> [IxB]-> RBM
learn _ rb [] = rb
learn rand rb iis = rb `R.deepSeqArray` learn r2 nrb (tail iis)
   where
      (r1,r2) = split rand
      nrb = batch r1 rb (head iis)
{-# INLINE learn #-}

-- given a batch of unbiased inputs, update the rbm weights from the batch at once
batch :: RandomGen r => r -> RBM -> IxB -> RBM
batch rand rb biased = uw
   where
      uw = R.computeUnboxedS $ R.zipWith (+) (weights rb) wd
      wd = weightUpdate rand rb biased
{-# INLINE batch #-}

-- given an unbiased input batch, generate the the RBM weight updates
weightUpdate :: RandomGen r => r -> RBM -> IxB -> HxI
weightUpdate rand rb biased = HxI $ R.computeUnboxedS $ R.zipWith (-) w1 w2
   where
      (r1,r2) = split rand
      hiddens = unHxB $ generate r1 rb biased
      newins = unBxI $ regenerate r2 rb hiddens
      w1 = hiddens `R.mmultS` biased
      w2 = hiddens `R.mmultS` newins
{-# INLINE weightUpdate #-}

-- given a biased input batch [(1:input)], generate a biased hidden layer sample batch
generate :: RandomGen r => r -> RBM -> IxB -> HxB
generate rand rb biased = HxB $ R.computeUnboxedS $ R.zipWith checkP hhs rands
   where
      hhs = unHxB $ hiddenProbs rb biased
      rands = unHxB $ randomArrayHxB (fst $ random rand) (R.extent hhs)
{-# INLINE generate #-}

-- given a batch of biased hidden layer samples, generate a batch of biased input layer samples
regenerate :: RandomGen r => r -> RBM -> BxH -> BxI
regenerate rand rb hidden = BxI $ R.computeUnboxedS $ R.zipWith checkP iis rands
   where
      iis = unBxI $ inputProbs hidden rb
      rands = unBxI $ randomArrayBxI (fst $ random rand) (R.extent iis)
{-# INLINE regenerate #-}

randomArrayBxI :: Int -> DIM2 -> BxI
randomArrayBxI seed sh = BxI $ R.traverse rands id set
   where
      rands = R.randomishDoubleArray sh 0 1 seed
      set _ (Z :. _ :. 0) = 0
      set ff sh = ff sh
{-# INLINE randomArrayBxI #-}

randomArrayHxB :: Int -> DIM2 -> HxB
randomArrayHxB seed sh = HxB $ R.traverse rands id set
   where
      rands = R.randomishDoubleArray sh 0 1 seed
      set _ (Z :. 0 :. _) = 0
      set ff sh = ff sh
{-# INLINE randomArrayHxB #-}

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
