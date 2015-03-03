module RBM(rbm
          ,learn
          ,batch
          ,energy
          ,test
          ,perf
          ) where

--benchmark modules
import Criterion.Main(defaultMainWith,defaultConfig,bgroup,bench,whnf)
import Criterion.Types(reportFile,timeLimit)
--test modules
import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess)
import Data.Word(Word8)

--impl modules
import Control.DeepSeq(NFData,rnf,deepseq)
import Data.List(transpose)
import System.Random(RandomGen
                    ,randomRs
                    ,split
                    ,mkStdGen
                    )

data RBM = RBM { weights :: [Double] -- weight matrix with 1 bias nodes in each layer, numHidden + 1 x numInputs  + 1
               , numInputs :: Int    -- size without bias node
               , _numHidden :: Int   -- size without bias node (not used atm)
               }

-- dethunk the lazy evaluation in batch learning
instance NFData RBM where
   rnf (RBM wws ni nh) = rnf (wws,ni,nh)

--create an rbm with some randomized weights
rbm :: RandomGen r => r -> Int -> Int -> RBM
rbm r ni nh = RBM nw ni nh
   where
      nw = take ((nh + 1)* (ni + 1)) $ randomRs (0,1) r

-- given a batch of unbiased inputs, update the RBM weights 
batch :: RandomGen r => r -> RBM -> [[Double]] -> RBM
batch _ rb [] = rb
batch rand rb iis =
   let (rr,nr) = split rand
       nrb = learn rr rb (head iis) 
   in  nrb `deepseq` batch nr nrb (tail iis)

-- given an rbm and an input, generate the energy
energy :: RBM -> [Double] -> Double
energy rb inputs = negate ee
   where
      biased = 1:inputs
      hhs = hiddenProbs rb biased
      wws = weights rb
      ee = wws `deepseq` hhs `deepseq` sum $ zipWith (*) wws [inp * hid | inp <- biased, hid <- hhs]

-- given an unbiased input, update the RBM weights
learn :: RandomGen r => r -> RBM -> [Double] -> RBM
learn rand rb inputs = rb { weights = uw }
   where
      (r1,r2) = split rand
      hiddens = generate r1 rb (1:inputs)
      newins = regenerate r2 rb hiddens
      w1 = vmult hiddens (1:inputs)
      w2 = vmult hiddens newins 
      wd = zipWith (-) w1 w2
      uw = zipWith (+) (weights rb) wd

-- given a biased input (1:input), generate a biased hidden layer sample
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
hiddenProbs rb biased = map (sigmoid . sum) $ groupByN ni $ zipWith (*) wws $ cycle biased
   where
      wws = weights rb
      ni = (numInputs rb) + 1

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
inputProbs rb hidden = map (sigmoid . sum) $ transpose $ groupByN ni $ zipWith (*) wws hhs
   where
      hhs = concat $ transpose $ replicate ni hidden
      wws = weights rb
      ni = (numInputs rb) + 1

--sample is 0 if generated number gg is less then probabiliy pp
applyP :: Double -> Double -> Double
applyP pp gg | pp < gg = 0
             | otherwise = 1

-- row vec * col vec
-- or (m x 1) * (1 x c) matrix multiply 
vmult :: [Double] -> [Double] -> [Double]
vmult xxs yys = [ (xx*yy) | xx <- xxs, yy<-yys]

-- split a list into lists of equal parts
-- groupByN 3 [1..] -> [[1,2,3],[4,5,6],[7,8,9]..]
groupByN :: Int -> [a] -> [[a]]
groupByN _ [] = []
groupByN n ls = (take n ls) : groupByN n (drop n ls)

-- sigmoid function
sigmoid :: Double -> Double
sigmoid d = 1 / (1 + (exp (negate d)))

-- tests

-- test to see if we can learn a random string
prop_learned :: Word8 -> Word8 -> Bool
prop_learned ni' nh' = (tail regened) == input
   where
      regened = regenerate (mr 2) lrb $ generate (mr 3) lrb (1:input)
      --learn the inputs
      lrb = batch (mr 1) rb inputs
      rb = rbm (mr 0) ni nh
      inputs = replicate 1000 $ input
      --convert a random list of its 0 to 1 to doubles
      input = map fromIntegral $ take ni $ randomRs (0::Int,1::Int) (mr 4)
      ni = fromIntegral ni' :: Int
      nh = fromIntegral nh' :: Int
      --creates a random number generator with a seed
      mr i = mkStdGen (ni + nh + i)


prop_learn :: Word8 -> Word8 -> Bool
prop_learn ni nh = ln == (length $ weights $ lrb)
   where
      ln = ((fi ni) + 1) * ((fi nh) + 1)
      lrb = learn' rb 1
      learn' rr ix = learn (mkStdGen ix) rr (take (fi ni) $ cycle [0,1])
      rb = rbm (mkStdGen 0) (fi ni) (fi nh)
      fi = fromIntegral

prop_batch :: Word8 -> Word8 -> Word8 -> Bool
prop_batch ix ni nh = ln == (length $ weights $ lrb)
   where
      ln = ((fi ni) + 1) * ((fi nh) + 1)
      lrb = batch rand rb inputs
      rb = rbm rand (fi ni) (fi nh)
      rand = mkStdGen ln
      inputs = replicate (fi ix) $ take (fi ni) $ cycle [0,1]
      fi = fromIntegral

prop_init :: Int -> Word8 -> Word8 -> Bool
prop_init gen ni nh = ((fi ni) + 1) * ((fi nh) + 1) == (length $ weights rb)
   where
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi = fromIntegral

prop_vmult :: Bool
prop_vmult = vmult [1,2,3] [4,5] == [1*4,1*5,2*4,2*5,3*4,3*5]

prop_hiddenProbs :: Int -> Word8 -> Word8 -> Bool
prop_hiddenProbs gen ni nh = (fi nh) + 1 == length pp
   where
      pp = hiddenProbs rb $ replicate ((fi ni) + 1) 0.0
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi = fromIntegral

prop_hiddenProbs2 :: Bool
prop_hiddenProbs2 = pp == map sigmoid [h0, h1]
   where
      h0 = w00 * i0 + w01 * i1 + w02 * i2  
      h1 = w10 * i0 + w11 * i1 + w12 * i2 
      i0:i1:i2:_ = [1..]
      w00:w01:w02:w10:w11:w12:_ = [1..]
      wws = [w00,w01,w02,w10,w11,w12]
      pp = hiddenProbs rb [i0,i1,i2]
      rb = RBM wws 2 1

prop_inputProbs :: Int -> Word8 -> Word8 -> Bool
prop_inputProbs gen ni nh = (fi ni) + 1 == length pp
   where
      pp = inputProbs rb $ replicate ((fi nh) + 1) 0.0
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi = fromIntegral

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
      rb = RBM wws 2 1

prop_energy :: Int -> Word8 -> Word8 -> Bool
prop_energy gen ni nh = not $ isNaN ee
   where
      ee = energy rb $ replicate (fi ni) 0.0
      rb = rbm (mkStdGen gen) (fi ni) (fi nh)
      fi = fromIntegral

test :: IO ()
test = do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 50 } 
       runtest p = check =<< verboseCheckWithResult cfg p 
   runtest prop_init
   runtest prop_energy
   runtest prop_hiddenProbs
   runtest prop_hiddenProbs2
   runtest prop_inputProbs
   runtest prop_inputProbs2
   runtest prop_vmult
   runtest prop_learn
   runtest prop_batch
   runtest prop_learned

perf :: IO ()
perf = do
   let file = "dist/perf-RBM.html"
       cfg = defaultConfig { reportFile = Just file, timeLimit = 0.5 }
   defaultMainWith cfg [
       bgroup "energy" [ bench "3x3"  $ whnf (prop_energy 0 3) 3
                       , bench "127x127"  $ whnf (prop_energy 0 127) 127
                       , bench "255x255"  $ whnf (prop_energy 0 255) 255
                       ]
      ,bgroup "hidden" [ bench "3x3"  $ whnf (prop_hiddenProbs 0 3) 3
                       , bench "127x127"  $ whnf (prop_hiddenProbs 0 127) 127
                       , bench "255x255"  $ whnf (prop_hiddenProbs 0 255) 255
                       ]
      ,bgroup "input" [ bench "3x3"  $ whnf (prop_inputProbs 0 3) 3
                      , bench "127x127"  $ whnf (prop_inputProbs 0 127) 127
                      , bench "255x255"  $ whnf (prop_inputProbs 0 255) 255
                      ]
      ,bgroup "batch" [ bench "15"  $ whnf (prop_batch 15 63) 63
                      , bench "63"  $ whnf (prop_batch 63 63) 63
                      , bench "127"  $ whnf (prop_batch 127 63) 63
                      , bench "255"  $ whnf (prop_batch 255 63) 63
                      ]
      ]
   putStrLn $ "perf log written to " ++ file
