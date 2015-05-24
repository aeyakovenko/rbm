{-# LANGUAGE BangPatterns #-}
module RBM.Repa(rbm
               ,learn
               ,energy
               ,generate
               ,regenerate
               ,Params(..)
               ,params
               ,BxI(..)
               ,BxH(..)
               ,HxB(..)
               ,IxH(..)
               ,HxI(..)
               ,IxB(..)
               ,RBM
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
                      ,Any(Any)
                      ,Z(Z)
                      ,(:.)((:.))
                      ,All(All)
                      ,(*^)
                      ,(+^)
                      ,(-^)
                      )
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Unsafe as Unsafe
import qualified Data.Array.Repa.Algorithms.Randomish as R
import qualified Data.Array.Repa.Algorithms.Matrix as R
import System.Random(RandomGen
                    ,random
                    ,randomRs
                    ,mkStdGen
                    ,split
                    )

import Control.Monad.Identity(runIdentity)
import Control.Monad(foldM)
import Control.DeepSeq(NFData, rnf)
import Debug.Trace(trace)


{--
 - weight matrix with 1 bias nodes in each layer, numHidden + 1 x numInputs  + 1
 --}
type RBM = HxI

{--
 - data types to keep track of matrix orientation
 -
 - H num hidden nodes
 - I num input nodes
 - B batch size
 --}
data HxI = HxI { unHxI :: (Array U DIM2 Double)}
data IxH = IxH { unIxH :: (Array U DIM2 Double)}
data BxI = BxI { unBxI :: (Array U DIM2 Double)}
data IxB = IxB { unIxB :: (Array U DIM2 Double)}
data HxB = HxB { unHxB :: (Array U DIM2 Double)}
data BxH = BxH { unBxH :: (Array U DIM2 Double)}

instance NFData HxI where
   rnf (HxI ar) = ar `R.deepSeqArray` ()

instance NFData BxI where
   rnf (BxI ar) = ar `R.deepSeqArray` ()

weights :: RBM -> HxI
weights wws = wws
{-# INLINE weights #-}

row :: DIM2 -> Int
row (Z :. r :. _) = r
{-# INLINE row #-}

col :: DIM2 -> Int
col (Z :. _ :. c) = c
{-# INLINE col #-}

type MSE = Double

data Params = Params { rate :: Double      -- rate of learning each input
                     , minMSE :: MSE       -- min MSE before done
                     , minEpochs :: Int    -- min number of times to repeat
                     , maxEpochs :: Int    -- max number of times to repeat
                     , seed :: Int         -- random seed
                     }

params :: Params
params = Params 0.001 0.01 1 100 0

--create an rbm with some randomized weights
rbm :: RandomGen r => r -> Int -> Int -> RBM
rbm r ni nh = HxI nw
   where
      nw = R.randomishDoubleArray (Z :. nh :. ni) (-0.01) (0.01) (fst $ random r)

infinity :: Double
infinity = read "Infinity"

-- update the rbm weights from each batch
learn :: (Monad m) => Params -> RBM -> [m BxI]-> m RBM
learn prm rb ins = do
   let rr = (mkStdGen $ seed prm)
       loop epoch crb [] _ r0 _ = loop (epoch + 1) crb ins infinity r0 0
       loop epoch crb _ _ _ _
         | epoch > (maxEpochs prm) = "maxepochs" `trace` return crb
       loop epoch crb _ mse _ _
         | epoch >= (minEpochs prm) && mse < (minMSE prm) = "minmse" `trace` return crb
       loop epoch crb bns _ r0 nmb
         | (nmb `mod` 10 == 0) = do
            let (r1,r2) = split r0
                rbatch = head $ drop (head $ randomRs (0::Int, (length ins)) r1) $ cycle ins
            rbatch' <- rbatch
            mse <- reconErr r1 crb [return rbatch']
            (show (mse, epoch, (length bns))) `trace` loop epoch crb bns mse r2 (nmb + 1)
       loop epoch crb bns mse r0 nmb = do
            let (r1,r2) = split r0
            nrb <- batch r1 (rate prm) crb [head bns]
            loop epoch nrb (tail bns) mse r2 (nmb + 1)
   loop 0 rb ins infinity rr (0::Int)
{-# INLINE learn #-}

{--
 - given an rbm and a biased input array, generate the energy
 --}
energy :: (Monad m) => RBM -> BxI -> m Double
energy rb bxi = do
   let wws = unHxI $ weights rb
   hxb <- (unHxB <$> hiddenProbs rb bxi)
   hxi <- hxb `mmultP` (unBxI bxi)
   enr <- (R.sumAllP $ wws *^ hxi)
   return $ negate enr

d2u :: (Monad m, R.Shape a) => Array D a Double -> m (Array U a Double)
d2u ar = R.computeP ar

{--
 - given a biased input generate probabilities of the hidden layer
 - incuding the biased probability
 -
 - map sigmoid $ biased `mmult` weights
 -
 --}
hiddenProbs :: (Monad m) => HxI -> BxI -> m HxB
hiddenProbs wws iis = do
   !hxb <- (unHxI wws) `mmultTP` (unBxI iis)
   HxB <$> (d2u $ R.map sigmoid hxb)
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
inputProbs :: (Monad m) => IxH -> BxH -> m IxB
inputProbs wws hhs = do
   !ixb <- (unIxH wws) `mmultTP` (unBxH hhs)
   IxB <$> (d2u $ R.map sigmoid ixb)
{-# INLINE inputProbs #-}


-- given a batch of unbiased inputs, update the rbm weights from the batch at once
-- dropouts...
-- 1 2 3   a d h    1a 2b 3c  1d 20 3f  1h 2i 3j
-- 4 5 6   b 0 i    4a 5b 6c  4d 50 6f  4h 5i 6j
--         c f j
--
-- zero out a BxI input column before generating the probs

batch :: (Monad m, RandomGen r) => r -> Double -> RBM -> [m BxI] -> m RBM
batch rand lrate hxi ins = foldM (weightUpdate rand lrate) hxi ins
{-# INLINE batch #-}

-- given an unbiased input batch, generate the the RBM weight updates
weightUpdate:: (Monad m, RandomGen r) => r -> Double -> HxI -> m BxI -> m (HxI)
weightUpdate rand lrate hxi mbxi = do
   bxi <- mbxi
   let cols = col $ R.extent $ unBxI bxi
       rows = row $ R.extent $ unBxI bxi
       loop rbm' (rr,rix) = do
            let vxi = R.slice (unBxI bxi) (Any :. (rix::Int) :. All)
            bxi' <- BxI <$> (d2u $ R.reshape (Z :. 1 :. cols) vxi)
            !wd <- unHxI <$> weightDiff rr rbm' bxi'
            !diffsum <- R.sumAllP $ R.map abs wd
            !weightsum <- R.sumAllP $ R.map abs (unHxI hxi)
            let lrate' = weightsum / diffsum * (lrate)
            let wd' = R.map ((*) lrate') wd
            HxI <$> (d2u $ (unHxI $ rbm') +^ wd')
   foldM loop hxi $ zip (splits rand) [0..(rows-1)]
{-# INLINE weightUpdate #-}

weightDiff :: (Monad m, RandomGen r) => r -> HxI -> BxI -> m HxI
weightDiff rand hxi bxi = do
   let (r1,r2) = split rand
   hxb <- generate r1 hxi bxi
   bxh <- BxH <$> (R.transpose2P (unHxB hxb))
   ixb <- IxB <$> (R.transpose2P (unBxI bxi))
   ixh <- IxH <$> (R.transpose2P (unHxI hxi))
   ixb' <- regenerate r2 ixh bxh
   w1 <- (unHxB hxb) `mmultTP` (unIxB ixb)
   w2 <- (unHxB hxb) `mmultTP` (unIxB ixb')
   HxI <$> (d2u $ w1 -^ w2)
{-# INLINE weightDiff #-}

reconErr :: (Monad m, RandomGen r) => r -> HxI -> [m BxI] -> m Double
reconErr rand hxi mbxis = do
   let loop !total mbxi = do
         bxi <- mbxi
         hxb <- generate rand hxi bxi
         bxh <- BxH <$> (R.transpose2P (unHxB hxb))
         ixh <- IxH <$> (R.transpose2P (unHxI hxi))
         ixb <- regenerate rand ixh bxh
         bxi' <- BxI <$> (R.transpose2P (unIxB ixb)) 
         wd <- d2u $ (unBxI bxi') -^ (unBxI bxi)
         !mse <- R.sumAllP $ R.map (\ xx -> xx ** 2) wd
         let sz = 1 + (row $ R.extent wd) * (col $ R.extent wd)
         return (total +  (mse / (fromIntegral sz)))
   !mse <- foldM loop 0 mbxis
   return $ (mse / (fromIntegral $ length mbxis))
{-# INLINE reconErr #-}

splits :: RandomGen r => r -> [r]
splits rp = rc : splits rn
   where
      (rc,rn) = split rp

-- given a biased input batch [(1:input)], generate a biased hidden layer sample batch
generate :: (Monad m, RandomGen r) => r -> HxI -> BxI -> m HxB
generate rand rb biased = do
   hhs <- unHxB <$> hiddenProbs rb biased
   rands <- unHxB <$> randomArrayHxB (fst $ random rand) (R.extent hhs)
   HxB <$> (d2u $ R.zipWith checkP hhs rands)
{-# INLINE generate #-}

-- given a batch of biased hidden layer samples, generate a batch of biased input layer samples
regenerate :: (Monad m, RandomGen r) => r -> IxH -> BxH -> m IxB
regenerate rand rb hidden = do
   iis <- unIxB <$> inputProbs rb hidden
   rands <- unIxB <$> (randomArrayIxB (fst $ random rand) (R.extent iis))
   IxB <$> (d2u $ R.zipWith checkP iis rands)
{-# INLINE regenerate #-}

randomArrayIxB :: (Monad m) => Int -> DIM2 -> m IxB
randomArrayIxB rseed sh = IxB <$> (d2u $ R.traverse rands id set)
   where
      rands = R.randomishDoubleArray sh 0 1 rseed
      set _ (Z :. _ :. 0) = 0
      set ff sh' = ff sh'
{-# INLINE randomArrayIxB #-}

randomArrayHxB :: (Monad m) => Int -> DIM2 -> m HxB
randomArrayHxB rseed sh = HxB <$> (d2u $ R.traverse rands id set)
   where
      rands = R.randomishDoubleArray sh 0 1 rseed
      set _ (Z :. 0 :. _) = 0
      set ff sh' = ff sh'
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

{--
 - matrix multiply
 - a x (transpose b)
 - based on mmultP from repa-algorithms-3.3.1.2
 -}
mmultTP  :: Monad m
        => Array U DIM2 Double
        -> Array U DIM2 Double
        -> m (Array U DIM2 Double)
mmultTP arr trr
 = [arr, trr] `R.deepSeqArrays`
   do
        let (Z :. h1  :. _) = R.extent arr
        let (Z :. w2  :. _) = R.extent trr
        R.computeP
         $ R.fromFunction (Z :. h1 :. w2)
         $ \ix   -> R.sumAllS
                  $ R.zipWith (*)
                        (Unsafe.unsafeSlice arr (Any :. (row ix) :. All))
                        (Unsafe.unsafeSlice trr (Any :. (col ix) :. All))
{-# NOINLINE mmultTP #-}

{--
 - regular matrix multiply
 - a x b
 - based on mmultP from repa-algorithms-3.3.1.2
 - basically moved the deepseq to seq the trr instead of brr
 -}
mmultP  :: Monad m
        => Array U DIM2 Double
        -> Array U DIM2 Double
        -> m (Array U DIM2 Double)
mmultP arr brr
 = do   trr <- R.transpose2P brr
        mmultTP arr trr
{-# NOINLINE mmultP #-}
-- test to see if we can learn a random string
run_prop_learned :: Double -> Int -> Int -> ([Double],[Double])
run_prop_learned lrate ni nh = runIdentity $ do
   let rb = rbm (mr 0) (fi ni) (fi nh)
       inputbatchL = concat $ replicate batchsz inputlst
       inputbatch = BxI $ R.fromListUnboxed (Z:. batchsz :.fi ni) $ inputbatchL
       inputarr = BxI $ R.fromListUnboxed (Z:. 1 :. fi ni) $ inputlst
       geninputs = randomRs (0::Int,1::Int) (mr 4)
       inputlst = map fromIntegral $ take (fi ni) $ 1:geninputs
       fi ww = 1 + ww
       mr i = mkStdGen (fi ni + fi nh + i)
       batchsz = 10
       par = params { rate = (rate params) * lrate }
   lrb <- learn par rb (replicate 100 (return inputbatch))
   hxb <- generate (mr 3) lrb inputarr
   ixh <- IxH <$> (R.transpose2P $ unHxI lrb)
   bxh <- BxH <$> (R.transpose2P $ unHxB hxb)
   ixb <- regenerate (mr 2) ixh bxh
   bxi <- BxI <$> (R.transpose2P $ unIxB ixb)
   return $ ((tail $ R.toList $ unBxI $ bxi), (tail $ R.toList $ unBxI $ inputarr))

prop_learned :: Word8 -> Word8 -> Bool
prop_learned ni nh = (uncurry (==)) $ run_prop_learned 1.0 (fi ni) (fi nh)
   where
      fi = fromIntegral

prop_not_learned :: Word8 -> Word8 -> Bool
prop_not_learned ni nh = (uncurry check) $ run_prop_learned (-1.0) (fi ni) (fi nh)
   where
      fi ii = fromIntegral ii
      check aas bbs = (null aas) || (aas /= bbs) || (minimum aas == 1.0)

prop_learn :: Word8 -> Word8 -> Bool
prop_learn ni nh = runIdentity $ do
   let inputs = R.fromListUnboxed (Z:.fi nh:.fi ni) $ take ((fi ni) * (fi nh)) $ cycle [0,1]
       rand = mkStdGen $ fi nh
       rb = rbm rand (fi ni) (fi nh)
       fi ww = 1 + (fromIntegral ww)
   lrb <- learn params rb [return $ BxI inputs]
   return $ (R.extent $ unHxI $ weights rb) == (R.extent $ unHxI $ weights $ lrb)

prop_batch :: Word8 -> Word8 -> Word8 -> Bool
prop_batch ix ni nh = runIdentity $ do
   let rb = rbm rand (fi ni) (fi nh)
       rand = mkStdGen $ fi ix
       inputs = R.fromListUnboxed (Z:.fi ix:.fi ni) $ take ((fi ni) * (fi ix)) $ cycle [0,1]
       fi ww = 1 + (fromIntegral ww)
   lrb <- batch rand 1.0 rb [return $ BxI inputs]
   return $ (R.extent $ unHxI $ weights rb) == (R.extent $ unHxI $ weights $ lrb)

prop_init :: Int -> Word8 -> Word8 -> Bool
prop_init gen ni nh = (fi ni) * (fi nh)  == (length $ R.toList $ unHxI $ weights rb)
   where
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi :: Word8 -> Int
      fi ww = 1 + (fromIntegral ww)

prop_hiddenProbs :: Int -> Word8 -> Word8 -> Bool
prop_hiddenProbs gen ni nh = runIdentity $ do
   let rb = rbm (mkStdGen gen) (fi ni) (fi nh)
       fi ww = 1 + (fromIntegral ww)
       input = BxI $ (R.randomishDoubleArray (Z :. 1 :. (fi ni)) 0 1 gen)
   pp <- hiddenProbs rb input
   return $ (fi nh) == (row $ R.extent $ unHxB pp)

prop_hiddenProbs2 :: Bool
prop_hiddenProbs2 = runIdentity $ do
   let h0 = w00 * i0 + w01 * i1 + w02 * i2
       h1 = w10 * i0 + w11 * i1 + w12 * i2
       i0:i1:i2:_ = [1..]
       w00:w01:w02:w10:w11:w12:_ = [1..]
       wws = [w00,w01,w02,w10,w11,w12]
       input = BxI $ R.fromListUnboxed (Z:.1:.3) $ [i0,i1,i2]
       rb = HxI $ R.fromListUnboxed (Z:.2:.3) $ wws
   pp <- R.toList <$> unHxB <$> hiddenProbs rb input
   return $ pp == map sigmoid [h0, h1]

prop_inputProbs :: Int -> Word8 -> Word8 -> Bool
prop_inputProbs gen ni nh = runIdentity $ do
   let hidden = BxH $ R.randomishDoubleArray (Z :. 1 :. (fi nh)) 0 1 gen
       rb = rbm (mkStdGen gen) (fi ni) (fi nh)
       fi ww = 1 + (fromIntegral ww)
   rb' <- IxH <$> (R.transpose2P (unHxI rb))
   pp <- unIxB <$> inputProbs rb' hidden
   return $ (fi ni) == (row $ R.extent pp)

prop_inputProbs2 :: Bool
prop_inputProbs2 = runIdentity $ do
   let i0 = w00 * h0 + w10 * h1
       i1 = w01 * h0 + w11 * h1
       i2 = w02 * h0 + w12 * h1
       h0:h1:_ = [1..]
       w00:w01:w02:w10:w11:w12:_ = [1..]
       wws = [w00,w01,w02,w10,w11,w12]
       hiddens = BxH $ R.fromListUnboxed (Z:.1:.2) [h0,h1]
       rb = HxI $ R.fromListUnboxed (Z:.2:.3) $ wws
   rb' <- IxH <$> (R.transpose2P (unHxI rb))
   pp <- inputProbs rb' hiddens
   pp' <- R.toList <$> R.transpose2P (unIxB pp)
   return $ pp' == map sigmoid [i0,i1,i2]

prop_energy :: Int -> Word8 -> Word8 -> Bool
prop_energy gen ni nh = runIdentity $ do
   let input = R.randomishDoubleArray (Z :. 1 :. (fi ni)) 0 1 gen
       rb = rbm (mkStdGen gen) (fi ni) (fi nh)
       fi ww = 1 + (fromIntegral ww)
   ee <- energy rb (BxI input)
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
   runtest "learned"      prop_learned
   runtest "notlearnred"  prop_not_learned

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
