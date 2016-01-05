module Test.Matrix(test) where

import Data.Matrix
import Prelude as P

import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess,maxSize)
import Control.Monad.Identity(runIdentity)
import qualified Data.Vector.Unboxed as V
import Data.Binary as B
import Control.DeepSeq as N



prop_splitRows :: Int -> Int -> Int -> Bool
prop_splitRows xx yy zz = (toList mm) == (concatMap toList splitted)
                       && num == (length splitted)
   where rr = abs xx + 1
         cc = abs yy + 1
         ss = abs zz + 1
         num = ceiling $ ((fromIntegral rr)::Double) / (fromIntegral ss)
         mm = fromList (rr,cc) $ P.map fromIntegral [1..(rr * cc)]
         splitted = runIdentity $ mapM d2u (splitRows ss mm)
   
prop_fold :: Bool
prop_fold = foldl (+) 0 (toList mm) == res
   where res = runIdentity $ fold (+) 0 mm
         mm = fromList (1,3) [1,2,3]

prop_unboxed :: Bool
prop_unboxed = (toUnboxed mm) == V.fromList [1,2,3]
   where mm = fromList (1,3) [1,2,3]

prop_binary :: Bool
prop_binary = (toList mm) == (toList $ B.decode $ B.encode $ mm)
   where mm = fromList (1,3) [1,2,3]

prop_cast :: Bool
prop_cast = (toList $ cast1 mm) == (toList $ cast2 mm)
   where mm = fromList (1,3) [1,2,3]

prop_show :: Bool
prop_show = 0 < (length $ show mm)
   where mm = fromList (1,3) [1,2,3]

prop_nfdata :: Bool
prop_nfdata = (toList mm) == (toList N.$!! mm)
   where mm = fromList (1,3) [1,2,3]

test :: IO ()
test =  do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 100, maxSize = 10 }
       runtest tst p =  do putStrLn tst; check =<< verboseCheckWithResult cfg p
   runtest "splitRows"  prop_splitRows
   runtest "fold"       prop_fold
   runtest "unboxed"    prop_unboxed
   runtest "binary"     prop_binary
   runtest "cast"       prop_cast
   runtest "show"       prop_show
   runtest "nfdata"     prop_nfdata
