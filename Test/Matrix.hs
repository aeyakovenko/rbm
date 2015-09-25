module Test.Matrix(test) where

import Data.Matrix
import Prelude as P

import System.Exit (exitFailure)
import Test.QuickCheck(verboseCheckWithResult)
import Test.QuickCheck.Test(isSuccess,stdArgs,maxSuccess,maxSize)
import Control.Monad.Identity(runIdentity)


prop_splitRows :: Int -> Int -> Int -> Bool
prop_splitRows xx yy zz = (toList mm) == (concatMap toList splitted)
                       && num == (length splitted)
   where rr = abs xx + 1
         cc = abs yy + 1
         ss = abs zz + 1
         num = ceiling $ ((fromIntegral rr)::Double) / (fromIntegral ss)
         mm = fromList (rr,cc) $ P.map fromIntegral [1..(rr * cc)]
         splitted = runIdentity $ mapM d2u (splitRows ss mm)
   
test :: IO ()
test =  do
   let check rr = if (isSuccess rr) then return () else exitFailure
       cfg = stdArgs { maxSuccess = 100, maxSize = 10 }
       runtest tst p =  do putStrLn tst; check =<< verboseCheckWithResult cfg p
   runtest "splitRows"  prop_splitRows
