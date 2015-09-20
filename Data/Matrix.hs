{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE ExistentialQuantification #-}
module Data.Matrix( Matrix(..)
                  , MatrixOps(..)
                  , R.U
                  , R.D
                  ) where

import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.Algorithms.Matrix as R
import qualified Data.Array.Repa.Unsafe as Unsafe
import qualified Data.Array.Repa.Algorithms.Randomish as R
import Control.DeepSeq(NFData, rnf)
import Data.Array.Repa(Array
                      ,U
                      ,D
                      ,DIM2
                      ,Any(Any)
                      ,Z(Z)
                      ,(:.)((:.))
                      ,All(All)
                      )

data Matrix d a b = R.Source d Double => Matrix (Array d DIM2 Double)

instance NFData (Matrix U a b) where
   rnf (Matrix ar) = ar `R.deepSeqArray` ()

class MatrixOps a b where
   mmult :: Monad m => (Matrix U a b) -> (Matrix U b c) -> m (Matrix U a c)
   mmultT :: Monad m => (Matrix U a b) -> (Matrix U c b) -> m (Matrix U a c)
   d2u :: Monad m => Matrix D a b -> m (Matrix U a b)
   (*^) :: Matrix c a b -> Matrix d a b -> (Matrix D a b)
   (+^) :: Matrix c a b -> Matrix d a b -> (Matrix D a b)
   (-^) :: Matrix c a b -> Matrix d a b -> (Matrix D a b)
   map :: (Double -> Double) -> Matrix c a b -> (Matrix D a b)
   cast1 :: Matrix c a b -> Matrix c d b
   cast2 :: Matrix c a b -> Matrix c a d
   transpose :: Monad m => Matrix U a b -> m (Matrix U b a)
   sum :: Monad m =>  Matrix c a b -> m Double
   elems :: Matrix c a b -> Int
   elems m = (row m) * (col m)
   {-# INLINE elems #-}
   row :: Matrix c a b -> Int
   col :: Matrix c a b -> Int
   shape :: Matrix c a b -> (Int,Int)
   shape m = (row m, col m)
   {-# INLINE shape #-}
   randomish :: (Int,Int) -> (Double,Double) -> Int -> Matrix U a b
   extractRows :: (Int,Int) -> Matrix c a b -> Matrix D a b 
   zipWith :: (Double -> Double -> Double) -> Matrix c a b -> Matrix c a b -> (Matrix D a b)
   fromList :: (Int,Int) -> [Double] -> Matrix U a b
   traverse :: (Double -> Int -> Int -> Double) -> Matrix c a b -> Matrix D a b
   toList :: Matrix U a b -> [Double]

instance MatrixOps a b where
   mmult (Matrix ab) (Matrix ba) = Matrix <$> (ab `mmultP` ba)
   mmultT (Matrix ab) (Matrix ab') = Matrix <$> (ab `mmultTP` ab')
   d2u (Matrix ar) = Matrix <$> (R.computeP ar)
   (Matrix ab) *^ (Matrix ab') = Matrix (ab R.*^ ab')
   {-# INLINE (*^) #-}
   (Matrix ab) +^ (Matrix ab') = Matrix (ab R.+^ ab')
   {-# INLINE (+^) #-}
   (Matrix ab) -^ (Matrix ab') = Matrix (ab R.-^ ab')
   {-# INLINE (-^) #-}
   map f (Matrix ar) = Matrix (R.map f ar)
   {-# INLINE map #-}
   cast1 (Matrix ar) = Matrix ar
   {-# INLINE cast1 #-}
   cast2 (Matrix ar) = Matrix ar
   {-# INLINE cast2 #-}
   transpose (Matrix ar) = Matrix <$> (R.transpose2P ar)
   sum (Matrix ar) = R.sumAllP ar
   row (Matrix ar) = (R.row (R.extent ar))
   {-# INLINE row #-}
   col (Matrix ar) = (R.col (R.extent ar))
   {-# INLINE col #-}
   randomish (r,c) (minv,maxv) seed = Matrix $ R.randomishDoubleArray (Z :. r :. c) minv maxv seed
   {-# INLINE randomish #-}
   extractRows (rix,num) mm@(Matrix ar) = Matrix $ R.extract
                                          (Z :. rix :. 0)
                                          (Z :. num :. (col mm))
                                          ar
   {-# INLINE extractRows #-}

   zipWith f (Matrix aa) (Matrix bb) = Matrix (R.zipWith f aa bb)
   {-# INLINE zipWith #-}
   fromList (r,c) lst = Matrix $ R.fromListUnboxed (Z:.r:.c) lst
   {-# INLINE fromList #-}
   traverse ff (Matrix ar) = Matrix $ R.traverse ar id func
         where func gv sh@(Z :. rr :. cc) = ff (gv sh) rr cc
   {-# INLINE traverse #-}
   toList (Matrix ab) = R.toList ab
   {-# INLINE toList #-}


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
                        (Unsafe.unsafeSlice arr (Any :. (R.row ix) :. All))
                        (Unsafe.unsafeSlice trr (Any :. (R.col ix) :. All))
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

