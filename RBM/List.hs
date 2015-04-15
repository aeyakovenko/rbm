module RBM.List(rbm
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
import Control.DeepSeq(NFData,rnf,deepseq)
import Data.List(transpose)
import Data.List.Split(chunksOf)
import System.Random(RandomGen
                    ,randomRs
                    ,split
                    ,mkStdGen
                    )

data RBM = RBM { weights :: [Double] -- weight matrix with 1 bias nodes in each layer, numHidden  x numInputs, which include the bias nodes
               , numInputs :: Int    -- size without bias node
               , numHidden :: Int    -- size without bias node
               }

-- dethunk the lazy evaluation in batch learning
instance NFData RBM where
   rnf (RBM wws ni nh) = rnf (wws,ni,nh)

--create an rbm with some randomized weights
rbm :: RandomGen r => r -> Int -> Int -> RBM
rbm _ ni nh
   | ni == 0 || nh == 0 = error "invalid number of nodes, must be at least 1"
rbm r ni nh = RBM nw ni nh
   where
      nw = take (nh * ni) $ randomRs (0,1) r

-- given an rbm and an input, generate the energy
energy :: RBM -> [Double] -> Double
energy rb inputs = negate ee
   where
      biased = inputs
      hhs = hiddenProbs rb biased
      wws = weights rb
      ee = hhs `deepseq` sum $ zipWith (*) wws [inp * hid | inp <- biased , hid <- hhs]

-- update the rbm weights from each batch
learn :: RandomGen r => r -> Double -> RBM -> [[[Double]]] -> RBM
learn _ _ rb [] = rb
learn rand rate rb iis = nrb `deepseq` learn r2 rate nrb (tail iis)
   where
      (r1,r2) = split rand
      nrb = batch r1 rate rb (head iis)

-- given a batch of biased inputs, update the rbm weights from the batch at once 
batch :: RandomGen r => r -> Double -> RBM -> [[Double]] -> RBM
batch rand rate rb inputs = rb { weights = uw }
   where
      uw = zipWith (+) (weights rb) wd'
      wd' = map ((*) rate) wd
      wd = weightUpdatesLoop rand rb inputs (take len [0..])
      len = (numInputs rb) * (numHidden rb)

-- given a batch of biased inputs, generate the the RBM weight updates for the batch
weightUpdatesLoop :: RandomGen r => r -> RBM -> [[Double]] -> [Double] -> [Double]
weightUpdatesLoop _ _ [] pw = pw
weightUpdatesLoop rand rb (inputs:rest) pw = weightUpdatesLoop r2 rb rest npw
   where
      npw = pw `deepseq` zipWith (+) pw wd
      wd = weightUpdate r1 rb inputs
      (r1,r2) = split rand

-- given an biased input, generate the the RBM weight updates
weightUpdate :: RandomGen r => r -> RBM -> [Double] -> [Double]
weightUpdate rand rb inputs = zipWith (-) w1 w2
   where
      (r1,r2) = split rand
      hiddens = generate r1 rb (inputs)
      newins = regenerate r2 rb hiddens
      w1 = tensor hiddens inputs
      w2 = tensor hiddens newins 

-- given a biased input generate a biased hidden layer sample
generate :: RandomGen r => r -> RBM -> [Double] -> [Double]
generate rand rb inputs = zipWith applyP (hiddenProbs rb inputs) (0:(randomRs (0,1) rand))

-- given a biased hidden layer sample, generate a biased input layer sample
regenerate :: RandomGen r => r -> RBM -> [Double] -> [Double]
regenerate rand rb hidden = zipWith applyP (inputProbs rb hidden) (0:(randomRs (0,1) rand))


{--
 - given a biased input generate probabilities of the hidden layer
 - incuding the biased probability
 -
 - basically does the following matrix x vector multiply
 - 
 -     w0 w1 w2
 - w0  00 01 02     i0     h0 = w00 * i0 + w01 * i1 + w02 * i2  
 - w1  10 11 12  x  i1     h1 = w10 * i0 + w11 * i1 + w12 * i2  
 -                  i2
--}
hiddenProbs :: RBM -> [Double] -> [Double]
hiddenProbs rb biased = map (sigmoid . sum) $ chunksOf ni $ zipWith (*) wws $ cycle biased
   where
      wws = weights rb
      ni = (numInputs rb)

{--
 - given a biased hidden sample generate probabilities of the input layer
 - incuding the biased probability
 -
 - transpose of the hiddenProbs function
 - 
 -     w0 w1 w2
 - w0  00 01 02 
 - w1  10 11 12 
 -        x
 -     h0 h1
 - 
 - i0 = w00 * h0 + w10 * h1
 - i1 = w01 * h0 + w11 * h1
 - i2 = w02 * h0 + w12 * h1
 - 
 --}
inputProbs :: RBM -> [Double] -> [Double]
inputProbs rb hidden = map (sigmoid . sum) $ transpose $ chunksOf ni $ zipWith (*) wws hhs
   where
      hhs = concat $ transpose $ replicate ni hidden
      wws = weights rb
      ni = (numInputs rb)

--sample is 0 if generated number gg is greater then probabiliy pp
--so the higher pp the more likely that it will generate a 1
applyP :: Double -> Double -> Double
applyP pp gg | pp < gg = 0
             | otherwise = 1

-- row vec * col vec
-- or (m x 1) * (1 x c) matrix multiply 
tensor :: [Double] -> [Double] -> [Double]
tensor xxs yys = [ (xx*yy) | xx <- xxs, yy<-yys]

-- sigmoid function
sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))

-- test to see if we can learn a random string
run_prop_learned :: Double -> Int -> Int -> Bool
run_prop_learned rate ni' nh' = (tail regened) == (tail input)
   where
      regened = regenerate (mr 2) lrb $ generate (mr 3) lrb input
      --learn the inputs
      lrb = learn (mr 1) rate rb [inputs]
      rb = rbm (mr 0) ni nh
      inputs = replicate 2000 $ input
      --convert a random list of its 0 to 1 to doubles
      isSame ls = maximum ls == minimum ls
      geninputs = head $ dropWhile isSame $ chunksOf ni $ randomRs (0::Int,1::Int) (mr 4)
      input = map fromIntegral $ take ni $ 1:geninputs
      ni = fi ni'
      nh = fi nh'
      --creates a random number generator with a seed
      mr i = mkStdGen (ni + nh + i)
      fi ii = 1 + (fromIntegral ii)

prop_learned :: Word8 -> Word8 -> Bool
prop_learned ni nh = run_prop_learned 1.0 (fi ni) (fi nh)
   where
      fi ii = (fromIntegral ii)

prop_notlearned :: Word8 -> Word8 -> Bool
prop_notlearned ni nh = not $ run_prop_learned (-1.0) (fi ni) (fi nh)
   where
      fi ii = 1 + (fromIntegral ii)


prop_learn :: Word8 -> Word8 -> Bool
prop_learn ni nh = ln == (length $ weights $ lrb)
   where
      ln = (fi ni) * (fi nh)
      lrb = learn' rb 1
      learn' rr ix = learn (mkStdGen ix) 1.0 rr [[(take (fi ni) $ cycle [0,1])]]
      rb = rbm (mkStdGen 0) (fi ni) (fi nh)
      fi ii = 1 + fromIntegral ii

prop_batch :: Word8 -> Word8 -> Word8 -> Bool
prop_batch ix ni nh = ln == (length $ weights $ lrb)
   where
      ln = (fi ni) * (fi nh)
      lrb = batch rand 1.0 rb inputs
      rb = rbm rand (fi ni) (fi nh)
      rand = mkStdGen ln
      inputs = replicate (fi ix) $ take (fi ni) $ cycle [0,1]
      fi ii = 1 + (fromIntegral ii)

prop_init :: Int -> Word8 -> Word8 -> Bool
prop_init gen ni nh = (fi ni) * (fi nh) == (length $ weights rb)
   where
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi ii = 1 + (fromIntegral ii)

prop_tensor :: Bool
prop_tensor = tensor [1,2,3] [4,5] == [1*4,1*5,2*4,2*5,3*4,3*5]

prop_hiddenProbs :: Int -> Word8 -> Word8 -> Bool
prop_hiddenProbs gen ni nh = (fi nh) == length pp
   where
      pp = hiddenProbs rb $ replicate (fi ni) 0.0
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi ii = 1 + (fromIntegral ii)

prop_hiddenProbs2 :: Bool
prop_hiddenProbs2 = pp == map sigmoid [h0, h1]
   where
      h0 = w00 * i0 + w01 * i1 + w02 * i2  
      h1 = w10 * i0 + w11 * i1 + w12 * i2 
      i0:i1:i2:_ = [1..]
      w00:w01:w02:w10:w11:w12:_ = [1..]
      wws = [w00,w01,w02,w10,w11,w12]
      pp = hiddenProbs rb [i0,i1,i2]
      rb = RBM wws 3 2

prop_inputProbs :: Int -> Word8 -> Word8 -> Bool
prop_inputProbs gen ni nh = (fi ni) == length pp
   where
      pp = inputProbs rb $ replicate (fi nh) 0.0
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi ii = 1 + (fromIntegral ii)

prop_inputProbs2 :: Bool
prop_inputProbs2 = pp == map sigmoid [i0,i1,i2]
   where
      i0 = w00 * h0 + w10 * h1
      i1 = w01 * h0 + w11 * h1
      i2 = w02 * h0 + w12 * h1
      h0:h1:_ = [1..]
      w00:w01:w02:w10:w11:w12:_ = [1..]
      wws = [w00,w01,w02,w10,w11,w12]
      pp = inputProbs rb [h0,h1]
      rb = RBM wws 3 2

prop_energy :: Int -> Word8 -> Word8 -> Bool
prop_energy gen ni nh = not $ isNaN ee
   where
      ee = energy rb $ take (fi ni) $ 1:(repeat 0.0)
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi ii = 1 + (fromIntegral ii)

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
   runtest "tensor"   prop_tensor
   runtest "learn"    prop_learn
   runtest "batch"    prop_batch
   runtest "learned"  prop_learned
   runtest "notlearned"  prop_notlearned

perf :: IO ()
perf = do
   let file = "dist/perf-list-RBM.html"
       cfg = defaultConfig { reportFile = Just file, timeLimit = 1.0 }
   defaultMainWith cfg [
       bgroup "energy" [ bench "3x3"  $ whnf (prop_energy 0 3) 3
                       , bench "63x63"  $ whnf (prop_energy 0 63) 63
                       , bench "127x127"  $ whnf (prop_energy 0 127) 127
                       ]
      ,bgroup "hidden" [ bench "3x3"  $ whnf (prop_hiddenProbs 0 3) 3
                       , bench "63x63"  $ whnf (prop_hiddenProbs 0 63) 63
                       , bench "127x127"  $ whnf (prop_hiddenProbs 0 127) 127
                       ]
      ,bgroup "input" [ bench "3x3"  $ whnf (prop_inputProbs 0 3) 3
                      , bench "63x63"  $ whnf (prop_inputProbs 0 63) 63
                      , bench "127x127"  $ whnf (prop_inputProbs 0 127) 127
                      ]
      ,bgroup "batch" [ bench "3"  $ whnf (prop_batch 3 3) 3
                      , bench "15"  $ whnf (prop_batch 15 15) 15
                      , bench "63"  $ whnf (prop_batch 63 63) 63
                      ]
      ]
   putStrLn $ "perf log written to " ++ file
